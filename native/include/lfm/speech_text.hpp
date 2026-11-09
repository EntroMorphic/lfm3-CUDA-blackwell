#pragma once
// Long-form TTS by sentences (#2 step 4, validation/speech-long-form-v1): the native text splitter and the
// per-sentence requests.
//
// Why: the backbone's sliding-attention layers (window 256) lose sight of the text being read about 180 frames
// into an utterance, and long utterances then skip and loop. One fresh request per sentence never gets there.
//
// split_sentences is tools/lfm3_speech_chunked_hf.py sentences(), exactly (tests/test_speech_text.py):
//   parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
//   a part under min_words words (str.split()) joins the next one; a short last part joins the one before.
// Whitespace is Python's str.isspace (the same set as re's Unicode \s), pinned code point by code point.
// sentence_requests turns a [system?, user] chat into one chat per sentence of the user text, each rendered
// later exactly as a single request is (lfm3_text::encode_messages).
#include <string>
#include <string_view>
#include <vector>

#include "lfm/lfm3_text.hpp"

namespace lfm::speech_text {

inline constexpr size_t kMinWords = 6;

// Python str.isspace(): bidirectional class WS, B or S, or general category Zs.
inline bool is_space(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

inline size_t count_words(std::string_view s) {
  size_t i = 0, words = 0;
  bool in_word = false;
  std::uint32_t c = 0;
  while (i < s.size()) {
    if (!lfm3_text::next_code_point(s, i, c)) throw std::runtime_error("speech text: invalid UTF-8");
    const bool sp = is_space(c);
    words += !sp && !in_word;
    in_word = !sp;
  }
  return words;
}

inline std::vector<std::string> split_sentences(std::string_view text, size_t min_words = kMinWords) {
  struct Cp { std::uint32_t c; size_t at; };
  std::vector<Cp> cps;
  for (size_t i = 0; i < text.size();) {
    const size_t at = i;
    std::uint32_t c = 0;
    if (!lfm3_text::next_code_point(text, i, c)) throw std::runtime_error("speech text: invalid UTF-8");
    cps.push_back({c, at});
  }
  const auto byte = [&](size_t k) { return k < cps.size() ? cps[k].at : text.size(); };
  size_t b = 0, e = cps.size();
  while (b < e && is_space(cps[b].c)) ++b;
  while (e > b && is_space(cps[e - 1].c)) --e;
  std::vector<std::string> parts;
  if (b == e) return parts;
  size_t start = b;
  for (size_t i = b + 1; i < e;) {  // cps[b] is not a space, so a cut always has a character before it
    const std::uint32_t prev = cps[i - 1].c;
    if (is_space(cps[i].c) && (prev == '.' || prev == '!' || prev == '?')) {
      size_t j = i;
      while (is_space(cps[j].c)) ++j;  // cps[e - 1] is not a space: the run ends inside the text
      parts.emplace_back(text.substr(byte(start), byte(i) - byte(start)));
      start = i = j;
    } else {
      ++i;
    }
  }
  parts.emplace_back(text.substr(byte(start), byte(e) - byte(start)));
  std::vector<std::string> out;
  for (auto& p : parts) {
    if (!out.empty() && count_words(out.back()) < min_words) out.back() += " " + p;
    else out.push_back(std::move(p));
  }
  if (out.size() > 1 && count_words(out.back()) < min_words) {
    out[out.size() - 2] += " " + out.back();
    out.pop_back();
  }
  return out;
}

// One chat per sentence of the user text: [system?, user] in, the same system and one sentence each out.
inline std::vector<std::vector<lfm3_text::Message>> sentence_requests(const std::vector<lfm3_text::Message>& messages) {
  const bool sys = messages.size() == 2 && messages[0].role == "system";
  if (!(messages.size() == 1 || sys) || messages.back().role != "user")
    throw std::runtime_error("speech split: messages must be [system, user] or [user]");
  const auto& user = messages.back();
  if (user.content_is_list) throw std::runtime_error("speech split: the user content must be a string");
  const auto sentences = split_sentences(user.content);
  if (sentences.empty()) throw std::runtime_error("speech split: the user text is empty");
  std::vector<std::vector<lfm3_text::Message>> out;
  for (const auto& s : sentences) {
    out.push_back(messages);
    out.back().back().content = s;
  }
  return out;
}

}  // namespace lfm::speech_text
