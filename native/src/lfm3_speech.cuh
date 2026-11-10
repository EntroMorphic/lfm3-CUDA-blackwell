#pragma once
// Text in -> text and speech out, end to end in the native engine (#2 step 4, journal/lfm3-tts-quality-bar.md).
// Include after lfm3_main.cu and lfm3_batch.cuh (LFM3_RUNTIME_ONLY, LFM3_NATIVE_GEMM).
//
// One request, HF `_generate` semantics (lfm/audio_modes.hpp holds the control flow, A4):
// - prefill: the validated Runner, with the mode's marker appended to the prompt;
// - text steps: greedy on the device over the language model's logits, with the mode machine's suppression set (HF
//   generates text with do_sample=False);
// - speech steps: the depth decoder (lfm/depth.cuh, A1/B1) on the language model's final-normed hidden state, the
//   frame sampler (lfm/audio_sampling.hpp, B2: temperature 0.4, penalty 1.2 over the last 64 accepted frames, a
//   random reveal order), the EOS rewrite, and the frame fed back as the sum of its 12 codebook rows (A3);
// - the language model's decode steps: lfm3_batch::BatchDecoder with one row;
// - emitted frames stream into the codec (lfm/codec.cuh, A2) 12 at a time; the waveform is 24 kHz FP32.
// Forced mode replays given tokens and frames instead of picking and sampling, and records the hidden state every
// frame step reads: the deterministic check of the whole loop.
#include <lfm/audio_encoder.cuh>
#include <lfm/audio_modes.hpp>
#include <lfm/speech_run.hpp>
#include <lfm/audio_sampling.hpp>
#include <lfm/codec.cuh>
#include <lfm/depth.cuh>

#include <chrono>
#include <cmath>
#include <deque>
#include <functional>

namespace lfm3_speech {

class Session {
 public:
  Session(Runner& runner, const std::string& model_dir)
      : model_dir_(model_dir), runner_(runner), dec_(runner, 1), depth_(runner.weights), codec_(model_dir + "/audio_tokenizer", 12),
        logits_dev_(size_t(lfm::depth::kCodebooks) * lfm::depth::kCodes), suppress_dev_(runner.s.vocabulary) {}

  Result generate(std::vector<int> prompt, const Options& o) {
    using namespace lfm::audio;
    // refused before any state changes: an empty prompt; an image, video or speech-output placeholder anywhere (this
    // engine takes no image or video input and no replayed frames, so HF would refuse or need features it does not
    // have); <|audio|> placeholders that are not exactly the given clips' spans (each clip's span is
    // <|vision_start|>, tokens(clip) placeholders, <|vision_end|>, in order, as the processor expands them); continuing
    // a codec stream that the previous call did not leave open (it finalized, or threw midway)
    if (prompt.empty()) throw std::runtime_error("speech: empty prompt");
    const int audio_token = ModeTokens{}.placeholders[0];
    for (int id : prompt)
      for (int p : ModeTokens{}.placeholders)
        if (id == p && p != audio_token)
          throw std::runtime_error("speech: image, video or speech-output placeholder in the prompt (not supported as input)");
    {
      std::vector<size_t> clip_tokens;
      for (const auto& clip : o.audio) {
        if (clip.empty()) throw std::runtime_error("speech: empty audio clip");
        clip_tokens.push_back(lfm::audio_features::tokens(clip.size()));
      }
      check_audio_spans(prompt, clip_tokens);
    }
    if (o.codec_continue && !stream_open_) throw std::runtime_error("speech: no open codec stream to continue");
    // the encoder (and its checkpoint contract) before any state changes
    if (!o.audio.empty() && !encoder_) encoder_ = std::make_unique<lfm3_audio_in::Encoder>(runner_.weights, model_dir_);
    stream_open_ = false;
    Result res;
    lfm::audio::ModeOptions mo;
    mo.mode = o.mode, mo.max_new_tokens = o.max_new_tokens, mo.return_audio_eos = o.return_audio_eos;
    ModeMachine m(ModeTokens{}, mo);
    const int marker = m.begin(prompt.back());
    if (marker >= 0) prompt.push_back(marker);
    res.prompt = prompt;
    const auto t0 = std::chrono::steady_clock::now();
    runner_.reset();
    if (o.audio.empty()) {
      runner_.forward(prompt);
    } else {
      const auto e0 = std::chrono::steady_clock::now();
      size_t total = 0;
      for (const auto& clip : o.audio) total += lfm::audio_features::tokens(clip.size());
      if (audio_capacity_ < total)
        audio_rows_ = std::make_unique<Buffer<bf16>>(total * size_t(runner_.s.hidden)), audio_capacity_ = total;
      size_t at = 0;
      for (const auto& clip : o.audio) {
        lfm::audio_features::Dump dump;
        const auto mel = lfm::audio_features::log_mel(clip);
        at += size_t(encoder_->encode(mel, audio_rows_->p + at * runner_.s.hidden, o.record_audio ? &dump : nullptr));
        if (o.record_audio) res.audio_dumps.push_back(std::move(dump));
      }
      if (at != total) throw std::runtime_error("speech: encoded audio rows differ from the placeholders");  // unreachable
      cuda_check(cudaDeviceSynchronize());
      res.encode_seconds = secs(e0);
      runner_.forward(prompt, audio_rows_->p, audio_token);
    }
    dec_.load(0);
    if (!o.codec_continue) codec_.reset();
    cuda_check(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    FrameSampler sampler({0.4, 1.2, 64, o.seed});
    dec_.embed_dump = o.record_hidden ? &res.embeds : nullptr;
    std::vector<float> host_logits(size_t(kCodebooks) * kCodes);
    std::vector<int32_t> pending;  // emitted frames not yet given to the codec
    size_t forced_token = 0, forced_frame = 0;
    std::vector<int> last_suppressed;
    for (Step st; (st = m.next()) != Step::Done;) {
      if (o.cancelled && o.cancelled()) throw Cancelled();
      Outcome r;
      if (st == Step::Text) {
        int token;
        if (!o.forced_tokens.empty() || !o.forced_frames.empty()) {
          if (forced_token >= o.forced_tokens.size()) throw std::runtime_error("speech: forced tokens exhausted");
          token = o.forced_tokens[forced_token++];
        } else {
          const auto sup = m.suppressed();
          if (sup != last_suppressed) {
            std::vector<unsigned char> mask(runner_.s.vocabulary, 0);
            for (int id : sup) mask[id] = 1;
            cuda_check(cudaMemcpy(suppress_dev_.p, mask.data(), mask.size(), cudaMemcpyHostToDevice));
            last_suppressed = sup;
          }
          const auto a = std::chrono::steady_clock::now();
          token = dec_.select(0.0, {0}, {0}, suppress_dev_.p)[0];
          res.text_pick_seconds += secs(a);
        }
        r = m.on_text(token);
        res.text_tokens.push_back(token);
        res.steps.push_back({false, r.decode, r.token, {}});
        if (r.emitted) {
          res.slots.push_back(r.slot);
          if (o.on_slot) o.on_slot(r.slot);
        }
        if (r.decode) {
          const auto a = std::chrono::steady_clock::now();
          dec_.step({r.token});
          cuda_check(cudaDeviceSynchronize());
          res.lm_step_seconds += secs(a);
          ++res.decode_steps;
        }
      } else {
        const bf16* h = dec_.hidden_state(0);
        if (o.record_hidden) {
          std::vector<uint16_t> bits(runner_.s.hidden);
          cuda_check(cudaMemcpy(bits.data(), h, bits.size() * 2, cudaMemcpyDeviceToHost));
          res.hidden.push_back(std::move(bits));
        }
        Frame raw;
        if (!o.forced_frames.empty() || !o.forced_tokens.empty()) {
          if (forced_frame >= o.forced_frames.size()) throw std::runtime_error("speech: forced frames exhausted");
          raw = o.forced_frames[forced_frame++];
        } else {
          const auto a = std::chrono::steady_clock::now();
          double in_depth = 0;
          depth_.begin_frame(h);
          raw = sampler.sample_frame([&](const int* codes, const bool* masked, const bool* picked, float* out) {
            const auto b = std::chrono::steady_clock::now();
            depth_.pass(codes, masked, picked, logits_dev_.p);
            cuda_check(cudaMemcpy(out, logits_dev_.p, host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
            in_depth += secs(b);
          });
          const double total = secs(a);
          res.depth_seconds += in_depth;
          res.sample_host_seconds += total - in_depth;
        }
        res.raw_frames.push_back(raw);
        r = m.on_frame(raw);
        res.steps.push_back({true, r.decode, -1, r.fed});
        if (!r.eos) sampler.accept(r.fed);
        if (r.emitted) {
          res.slots.push_back(r.slot);
          if (o.on_slot) o.on_slot(r.slot);
          if (!r.eos) {  // the codec takes speech frames only
            res.frames.push_back(r.fed);
            pending.insert(pending.end(), r.fed.begin(), r.fed.end());
            if (pending.size() == 12 * 12) {
              const auto a = std::chrono::steady_clock::now();
              const size_t before = res.wav.size();
              codec_.decode(pending.data(), 12, false, res.wav), pending.clear();
              res.codec_seconds += secs(a);
              if (o.on_audio && res.wav.size() > before) o.on_audio(res.wav.data() + before, res.wav.size() - before);
            }
          }
        }
        if (r.decode) {
          std::vector<int> codes(r.fed.begin(), r.fed.end());
          const auto a = std::chrono::steady_clock::now();
          dec_.step({ModeTokens{}.audio_out}, &codes);
          cuda_check(cudaDeviceSynchronize());
          res.lm_step_seconds += secs(a);
          ++res.decode_steps;
        }
      }
    }
    dec_.embed_dump = nullptr;
    if ((!o.forced_tokens.empty() || !o.forced_frames.empty()) &&
        (forced_token != o.forced_tokens.size() || forced_frame != o.forced_frames.size()))
      throw std::runtime_error("speech: forced replay ended with tokens or frames unused");
    const size_t before = res.wav.size();
    codec_.decode(pending.empty() ? nullptr : pending.data(), int(pending.size() / 12), o.codec_finalize, res.wav);
    cuda_check(cudaDeviceSynchronize());
    stream_open_ = !o.codec_finalize;
    if (o.on_audio && res.wav.size() > before) o.on_audio(res.wav.data() + before, res.wav.size() - before);
    const auto t2 = std::chrono::steady_clock::now();
    res.prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
    res.generate_seconds = std::chrono::duration<double>(t2 - t1).count();
    return res;
  }

 private:
  static double secs(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
  }
  std::string model_dir_;
  std::unique_ptr<lfm3_audio_in::Encoder> encoder_;
  std::unique_ptr<Buffer<bf16>> audio_rows_;
  size_t audio_capacity_ = 0;
  Runner& runner_;
  lfm3_batch::BatchDecoder dec_;
  lfm::depth::Decoder depth_;
  lfm::codec::Decoder codec_;
  lfm::fp32::DeviceArray<float> logits_dev_;
  Buffer<unsigned char> suppress_dev_;
  bool stream_open_ = false;  // the last call returned normally with codec_finalize = false
};

// (run_chunks and wav_bytes, used by lfm3-speech and liquid-lfm3-server alike, are in lfm/speech_run.hpp)
}  // namespace lfm3_speech
