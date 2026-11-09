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
#include <lfm/audio_modes.hpp>
#include <lfm/audio_sampling.hpp>
#include <lfm/codec.cuh>
#include <lfm/depth.cuh>

#include <chrono>
#include <deque>

namespace lfm3_speech {

struct Options {
  lfm::audio::Requested mode = lfm::audio::Requested::Audio;
  int max_new_tokens = 512;
  uint64_t seed = 0;
  bool return_audio_eos = false;
  // forced replay: the text tokens and the raw frames to use, in order (empty: pick and sample)
  std::vector<int> forced_tokens;
  std::vector<lfm::audio::Frame> forced_frames;
  bool record_hidden = false;
  // long-form TTS by sentences (tools/lfm3_speech_chunked_hf.py): each sentence is its own request, and the codec
  // stream runs on across them: codec_continue skips the reset, codec_finalize = false keeps the stream's tail open
  bool codec_continue = false, codec_finalize = true;
};

struct Result {
  std::vector<int> prompt, slots, text_tokens;
  std::vector<lfm::audio::Frame> frames;    // emitted (codec input)
  std::vector<lfm::audio::Frame> raw_frames;  // as sampled, EOS frames included
  std::vector<float> wav;
  std::vector<std::vector<uint16_t>> hidden;  // per frame step, the BF16 bits of the state the depth decoder read
  std::vector<std::vector<uint16_t>> embeds;  // per decoded step (record_hidden), the BF16 input embedding fed
  struct StepRecord { bool frame, decode; int token; lfm::audio::Frame fed; };
  std::vector<StepRecord> steps;
  int decode_steps = 0;
  double prefill_seconds = 0, generate_seconds = 0;
  // where the generation time goes (host wall time, each phase synchronized at its end)
  double lm_step_seconds = 0, text_pick_seconds = 0, depth_seconds = 0, sample_host_seconds = 0, codec_seconds = 0;
};

class Session {
 public:
  Session(Runner& runner, const std::string& model_dir)
      : runner_(runner), dec_(runner, 1), depth_(runner.weights), codec_(model_dir + "/audio_tokenizer", 12),
        logits_dev_(size_t(lfm::depth::kCodebooks) * lfm::depth::kCodes), suppress_dev_(runner.s.vocabulary) {}

  Result generate(std::vector<int> prompt, const Options& o) {
    using namespace lfm::audio;
    Result res;
    lfm::audio::ModeOptions mo;
    mo.mode = o.mode, mo.max_new_tokens = o.max_new_tokens, mo.return_audio_eos = o.return_audio_eos;
    ModeMachine m(ModeTokens{}, mo);
    const int marker = m.begin(prompt.back());
    if (marker >= 0) prompt.push_back(marker);
    res.prompt = prompt;
    const auto t0 = std::chrono::steady_clock::now();
    runner_.reset();
    runner_.forward(prompt);
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
        if (r.emitted) res.slots.push_back(r.slot);
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
          if (!r.eos) {  // the codec takes speech frames only
            res.frames.push_back(r.fed);
            pending.insert(pending.end(), r.fed.begin(), r.fed.end());
            if (pending.size() == 12 * 12) {
              const auto a = std::chrono::steady_clock::now();
              codec_.decode(pending.data(), 12, false, res.wav), pending.clear();
              res.codec_seconds += secs(a);
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
    codec_.decode(pending.empty() ? nullptr : pending.data(), int(pending.size() / 12), o.codec_finalize, res.wav);
    cuda_check(cudaDeviceSynchronize());
    const auto t2 = std::chrono::steady_clock::now();
    res.prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
    res.generate_seconds = std::chrono::duration<double>(t2 - t1).count();
    return res;
  }

 private:
  static double secs(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
  }
  Runner& runner_;
  lfm3_batch::BatchDecoder dec_;
  lfm::depth::Decoder depth_;
  lfm::codec::Decoder codec_;
  lfm::fp32::DeviceArray<float> logits_dev_;
  Buffer<unsigned char> suppress_dev_;
};

}  // namespace lfm3_speech
