// lfm3-speech: text in -> text and speech out, natively (#2 step 4; src/lfm3_speech.cuh).
//
//   lfm3-speech MODEL_DIR REQUEST.json OUT_STEM
// REQUEST: {"prompt_ids": [...], "mode": "audio" | "interleaved" | "text" | "none", "max_new_tokens": N, "seed": S,
//           "capacity": C (default 4096), "forced_tokens": [...], "forced_frames": [[12 codes]...],
//           "record_hidden": false, "chunks": [[ids...], ...]}
//      or, rendered natively: {"messages": [...], "split": false, ...}: the chat (lfm/lfm3_text.hpp, the text
//      binary's "messages" contract) instead of prompt_ids; with "split": true, [system, user] or [user] is read
//      sentence by sentence (lfm/speech_text.hpp), each sentence one chunk as below, and the chunk json adds "text".
// "chunks" (long-form TTS by sentences): each entry is one sentence's whole prompt, generated as its own request
// with seed SEED * 1000 + k; the codec stream runs on across them, so OUT_STEM.wav is one stream. The outputs
// concatenate the chunks (prompt: the first chunk's), and the json adds "chunks": [{frames, slots, ended_by_eos}].
// A chunk "ended_by_eos" as tools/lfm3_speech_chunked_hf.py decides it: its last sampled frame is the EOS frame and
// its last slot is <|im_end|>. "slots" follow HF's default return_audio_eos=False (an EOS frame emits no slot); the
// chunked HF tool passes True, so its summary counts one more slot per EOS frame.
// Refused before the model loads or anything is generated: an unknown mode, an empty prompt or chunk, and a prompt
// that cannot fit max_new_tokens in the capacity. Earlier outputs of OUT_STEM are removed first, so a failed run
// leaves none of them behind looking current; the exit status is nonzero on any failure, including a failed write.
// OUT_STEM.json: prompt (with the mode marker), slots, text tokens, frame counts, timings (real_time_factor: audio
// over generate_seconds, the decode loop; real_time_factor_end_to_end adds the prefills);
// OUT_STEM.frames.i32 (emitted frames, [frame][12]); OUT_STEM.raw.i32 (sampled frames, EOS frame included);
// OUT_STEM.wav (24 kHz 16-bit PCM, for listening) and OUT_STEM.f32 (the exact FP32 waveform);
// OUT_STEM.hidden.bf16 when record_hidden ([frame step][2048], the state the depth decoder read), and
// OUT_STEM.embeds.bf16 ([decoded step][2048], the input embedding fed). steps: [frame, decoded, token, fed codes...].
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"
#include "../src/lfm3_speech.cuh"
#include <lfm/speech_text.hpp>

#include <cstdio>

namespace {
void write_wav(const std::string& path, const std::vector<float>& x, int rate) {
  std::ofstream f(path, std::ios::binary);
  auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  const uint32_t bytes = uint32_t(x.size() * 2);
  f.write("RIFF", 4); u32(36 + bytes); f.write("WAVEfmt ", 8); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2);
  u16(2); u16(16); f.write("data", 4); u32(bytes);
  for (float v : x) {
    const float c = std::max(-1.0f, std::min(1.0f, v));
    const int16_t s = int16_t(std::lrint(c * 32767.0f));
    f.write(reinterpret_cast<const char*>(&s), 2);
  }
  if (!f) throw std::runtime_error("cannot write " + path);
}
template <class T>
void write_raw(const std::string& path, const T* p, size_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(p), std::streamsize(n * sizeof(T)));
  if (!f) throw std::runtime_error("cannot write " + path);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4) throw std::runtime_error("usage: lfm3-speech MODEL_DIR REQUEST.json OUT_STEM");
  const auto req = read_json(argv[2]);
  const std::string stem = argv[3];
  const std::string mode = req.has("mode") ? req.at("mode").string() : "audio";
  if (mode != "audio" && mode != "interleaved" && mode != "text" && mode != "none")
    throw std::runtime_error("mode must be audio, interleaved, text or none");
  for (const char* ext : {".wav", ".f32", ".frames.i32", ".raw.i32", ".hidden.bf16", ".embeds.bf16", ".json"})
    std::filesystem::remove(stem + ext);
  const int capacity = req.has("capacity") ? req.at("capacity").integer() : 4096;
  auto cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", capacity);
  Runner runner(argv[1], cfg);
  lfm3_speech::Session session(runner, argv[1]);
  lfm3_speech::Options o;
  using lfm::audio::Requested;
  o.mode = mode == "audio" ? Requested::Audio : mode == "interleaved" ? Requested::Interleaved
         : mode == "text" ? Requested::Text : Requested::None;
  o.max_new_tokens = req.at("max_new_tokens").integer();
  o.seed = req.has("seed") ? uint64_t(req.at("seed").integer()) : 0;
  o.record_hidden = req.has("record_hidden") && req.at("record_hidden").boolean();
  if (req.has("forced_tokens")) for (auto& x : req.at("forced_tokens").array()) o.forced_tokens.push_back(x.integer());
  if (req.has("forced_frames"))
    for (auto& f : req.at("forced_frames").array()) {
      lfm::audio::Frame fr{};
      const auto& a = f.array();
      if (a.size() != 12) throw std::runtime_error("a forced frame needs 12 codes");
      for (int c = 0; c < 12; ++c) fr[c] = a[c].integer();
      o.forced_frames.push_back(fr);
    }
  std::vector<std::vector<int>> prompts;
  std::vector<std::string> chunk_texts;
  const bool split = req.has("split") && req.at("split").boolean();
  const bool chunked = req.has("chunks") || split;
  if (chunked && (!o.forced_tokens.empty() || !o.forced_frames.empty())) throw std::runtime_error("chunks cannot be forced");
  if (req.has("messages")) {
    if (req.has("prompt_ids") || req.has("chunks")) throw std::runtime_error("messages cannot be combined with prompt_ids or chunks");
    const std::string path = std::string(argv[1]) + "/tokenizer.json";
    const auto tok = lfm::lfm3_text::load_checked(path, lfm::tokenizer_token_count(path, cfg.shape.vocabulary));
    const auto messages = lfm::lfm3_text::parse_messages(req.at("messages"));
    const auto chats = split ? lfm::speech_text::sentence_requests(messages)
                             : std::vector<std::vector<lfm::lfm3_text::Message>>{messages};
    for (const auto& c : chats) {
      prompts.push_back(lfm::lfm3_text::encode_messages(*tok, c, capacity, o.max_new_tokens));
      if (split) chunk_texts.push_back(c.back().content);
    }
  } else if (split) {
    throw std::runtime_error("split needs messages");
  } else if (req.has("chunks")) {
    for (auto& c : req.at("chunks").array()) {
      prompts.emplace_back();
      for (auto& x : c.array()) prompts.back().push_back(x.integer());
    }
    if (prompts.empty()) throw std::runtime_error("chunks is empty");
  } else {
    prompts.emplace_back();
    for (auto& x : req.at("prompt_ids").array()) prompts.back().push_back(x.integer());
  }
  for (const auto& p : prompts) {  // (the messages path has checked this already, in encode_messages)
    if (p.empty()) throw std::runtime_error("empty prompt");
    if (o.max_new_tokens < 1 || p.size() + size_t(o.max_new_tokens) > size_t(capacity))
      throw std::runtime_error("prompt (" + std::to_string(p.size()) + ") + max_new_tokens exceeds capacity (" +
                               std::to_string(capacity) + ")");
  }
  lfm3_speech::Result r;
  struct ChunkInfo { size_t frames, slots; bool eos; };
  std::vector<ChunkInfo> chunk_info;
  const uint64_t seed = o.seed;
  for (size_t k = 0; k < prompts.size(); ++k) {
    if (chunked) o.seed = seed * 1000 + k;
    o.codec_continue = k > 0, o.codec_finalize = k + 1 == prompts.size();
    auto c = session.generate(prompts[k], o);
    const auto& eos_frame = c.raw_frames.empty() ? lfm::audio::Frame{} : c.raw_frames.back();
    const bool ended = !c.raw_frames.empty() &&
                       std::find(eos_frame.begin(), eos_frame.end(), lfm::audio::kEosCode) != eos_frame.end() &&
                       !c.slots.empty() && c.slots.back() == lfm::audio::ModeTokens{}.eos[0];
    chunk_info.push_back({c.frames.size(), c.slots.size(), ended});
    if (k == 0) {
      r = std::move(c);
      continue;
    }
    auto cat = [](auto& to, auto& from) { to.insert(to.end(), from.begin(), from.end()); };
    cat(r.slots, c.slots), cat(r.text_tokens, c.text_tokens), cat(r.frames, c.frames), cat(r.raw_frames, c.raw_frames);
    cat(r.wav, c.wav), cat(r.hidden, c.hidden), cat(r.embeds, c.embeds), cat(r.steps, c.steps);
    r.decode_steps += c.decode_steps;
    r.prefill_seconds += c.prefill_seconds, r.generate_seconds += c.generate_seconds;
    r.lm_step_seconds += c.lm_step_seconds, r.text_pick_seconds += c.text_pick_seconds;
    r.depth_seconds += c.depth_seconds, r.sample_host_seconds += c.sample_host_seconds, r.codec_seconds += c.codec_seconds;
  }
  write_wav(stem + ".wav", r.wav, 24000);
  write_raw(stem + ".f32", r.wav.data(), r.wav.size());
  std::vector<int32_t> fr, raw;
  for (auto& f : r.frames) fr.insert(fr.end(), f.begin(), f.end());
  for (auto& f : r.raw_frames) raw.insert(raw.end(), f.begin(), f.end());
  write_raw(stem + ".frames.i32", fr.data(), fr.size());
  write_raw(stem + ".raw.i32", raw.data(), raw.size());
  if (o.record_hidden) {
    std::vector<uint16_t> h;
    for (auto& x : r.hidden) h.insert(h.end(), x.begin(), x.end());
    write_raw(stem + ".hidden.bf16", h.data(), h.size());
    std::vector<uint16_t> e;
    for (auto& x : r.embeds) e.insert(e.end(), x.begin(), x.end());
    write_raw(stem + ".embeds.bf16", e.data(), e.size());
  }
  std::ofstream f(stem + ".json");
  f << "{\"mode\":" << lfm::lfm3_text::json_string(mode) << ",\"prompt\":";
  print_ids(f, r.prompt);
  f << ",\"slots\":"; print_ids(f, r.slots);
  f << ",\"text_tokens\":"; print_ids(f, r.text_tokens);
  f << ",\"steps\":[";
  for (size_t i = 0; i < r.steps.size(); ++i) {
    const auto& s = r.steps[i];
    f << (i ? "," : "") << "[" << (s.frame ? 1 : 0) << "," << (s.decode ? 1 : 0) << "," << s.token;
    if (s.frame) for (int c : s.fed) f << "," << c;
    f << "]";
  }
  f << "]";
  if (chunked) {
    f << ",\"chunks\":[";
    for (size_t k = 0; k < chunk_info.size(); ++k)
      f << (k ? "," : "") << "{\"frames\":" << chunk_info[k].frames << ",\"slots\":" << chunk_info[k].slots
        << ",\"ended_by_eos\":" << (chunk_info[k].eos ? "true" : "false")
        << (chunk_texts.empty() ? "" : ",\"text\":" + lfm::lfm3_text::json_string(chunk_texts[k])) << "}";
    f << "]";
  }
  f << ",\"frames\":" << r.frames.size() << ",\"raw_frames\":" << r.raw_frames.size() << ",\"samples\":" << r.wav.size()
    << ",\"audio_seconds\":" << r.wav.size() / 24000.0 << ",\"decode_steps\":" << r.decode_steps
    << ",\"prefill_seconds\":" << r.prefill_seconds << ",\"generate_seconds\":" << r.generate_seconds
    << ",\"lm_step_seconds\":" << r.lm_step_seconds << ",\"text_pick_seconds\":" << r.text_pick_seconds
    << ",\"depth_seconds\":" << r.depth_seconds << ",\"sample_host_seconds\":" << r.sample_host_seconds
    << ",\"codec_seconds\":" << r.codec_seconds
    << ",\"real_time_factor\":" << (r.wav.empty() ? 0.0 : (r.wav.size() / 24000.0) / r.generate_seconds)
    << ",\"real_time_factor_end_to_end\":"
    << (r.wav.empty() ? 0.0 : (r.wav.size() / 24000.0) / (r.prefill_seconds + r.generate_seconds)) << "}\n";
  f.close();
  if (!f) throw std::runtime_error("cannot write " + stem + ".json");
  std::cout << "{\"frames\":" << r.frames.size() << ",\"audio_seconds\":" << r.wav.size() / 24000.0
            << ",\"generate_seconds\":" << r.generate_seconds << ",\"slots\":" << r.slots.size() << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-speech: " << e.what() << "\n";
  return 1;
}
