#pragma once
// Long-form TTS by sentences (#2 step 4, validation/speech-long-form-v1): the native text splitter and the
// per-sentence requests.
//
// Why: the backbone's sliding-attention layers (window 256) lose sight of the text being read about 180 frames
// into an utterance, and long utterances then skip and loop. One fresh request per sentence never gets there.
//
// split_sentences is tools/lfm3_speech_chunked_hf.py sentences(), exactly (tests/test_speech_text.py); the rule was
// chosen by test (validation/speech-split-v1: WER, then Tripp's listening):
//   1. cut the stripped text at each whitespace run that follows . ! or ?, except after a title (Mr. Dr. ...: never)
//      or after an abbreviation (a.m. U.S. Jan. ...) when the next word does not start with an ASCII capital;
//   2. a part under kMinWords words (str.split()) joins the next one; a short last part joins the one before;
//   3. a chunk over kCapWords words is cut at its last clause mark (a word ending in , ; : or an em or en dash, or
//      followed by a lone - em or en dash) that leaves both sides kMinWords long and the first at most kCapWords;
//      with none, into the fewest equal pieces; repeated. Cut pieces are their words joined by single spaces.
// Whitespace is Python's str.isspace (the same set as re's Unicode \s), pinned code point by code point.
// sentence_requests turns a [system?, user] chat into one chat per sentence of the user text, each rendered
// later exactly as a single request is (lfm3_text::encode_messages).
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "lfm/lfm3_text.hpp"

namespace lfm::speech_text {

inline constexpr size_t kMinWords = 6;
inline constexpr size_t kCapWords = 49;  // the longest single chunk that read cleanly on every seed (speech-split-v1)

inline bool in_list(std::string_view word, std::initializer_list<std::string_view> list) {
  std::string low(word);
  for (char& c : low) if (c >= 'A' && c <= 'Z') c = char(c + 32);
  for (auto w : list) if (low == w) return true;
  return false;
}
inline bool is_title(std::string_view w) { return in_list(w, {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "mt.", "no."}); }
inline bool is_abbreviation(std::string_view w) {
  return in_list(w, {"a.m.", "p.m.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "vs.", "jan.", "feb.", "mar.", "apr.", "jun.",
                     "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."});
}

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

// The words of s (maximal runs of non-whitespace), as views into s.
inline std::vector<std::string_view> split_words(std::string_view s) {
  std::vector<std::string_view> words;
  size_t i = 0, word = std::string_view::npos;
  while (i < s.size()) {
    const size_t at = i;
    std::uint32_t c = 0;
    if (!lfm3_text::next_code_point(s, i, c)) throw std::runtime_error("speech text: invalid UTF-8");
    if (is_space(c)) {
      if (word != std::string_view::npos) words.push_back(s.substr(word, at - word)), word = std::string_view::npos;
    } else if (word == std::string_view::npos) {
      word = at;
    }
  }
  if (word != std::string_view::npos) words.push_back(s.substr(word));
  return words;
}

inline std::string join_words(const std::vector<std::string_view>& w, size_t from, size_t to) {
  std::string out;
  for (size_t k = from; k < to; ++k) (out += (k > from ? " " : "")) += w[k];
  return out;
}

inline std::uint32_t last_code_point(std::string_view w) {
  size_t i = w.size() - 1;
  while (i > 0 && (static_cast<unsigned char>(w[i]) & 0xC0) == 0x80) --i;
  std::uint32_t c = 0;
  lfm3_text::next_code_point(w, i, c);
  return c;
}

inline void cap_comma(const std::string& chunk, size_t cap, std::vector<std::string>& out) {
  const auto w = split_words(chunk);
  if (w.size() <= cap) { out.push_back(chunk); return; }
  size_t cut = 0;
  for (size_t i = 0; i + 1 < w.size(); ++i) {
    const std::uint32_t last = last_code_point(w[i]);
    const bool mark = last == ',' || last == ';' || last == ':' || last == 0x2014 || last == 0x2013 ||
                      w[i + 1] == "-" || w[i + 1] == "\u2014" || w[i + 1] == "\u2013";
    if (mark && i + 1 >= kMinWords && i + 1 <= cap && w.size() - (i + 1) >= kMinWords) cut = i + 1;
  }
  if (cut == 0) {  // the fewest equal pieces of at most cap words
    const size_t pieces = (w.size() + cap - 1) / cap;
    cut = (w.size() + pieces - 1) / pieces;
  }
  out.push_back(join_words(w, 0, cut));
  cap_comma(join_words(w, cut, w.size()), cap, out);
}

inline std::vector<std::string> split_sentences(std::string_view text, size_t min_words = kMinWords,
                                                size_t cap = kCapWords) {
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
  size_t start = b, prev_end = b;
  for (size_t i = b + 1; i < e;) {  // cps[b] is not a space, so a run always has a word before it
    if (!is_space(cps[i].c)) { ++i; continue; }
    size_t j = i;
    while (is_space(cps[j].c)) ++j;  // cps[e - 1] is not a space: the run ends inside the text
    const std::string_view before = text.substr(byte(prev_end), byte(i) - byte(prev_end));
    const std::uint32_t prev = cps[i - 1].c, next = cps[j].c;
    prev_end = j;
    if ((prev == '.' || prev == '!' || prev == '?') && !is_title(before) &&
        !(is_abbreviation(before) && !(next >= 'A' && next <= 'Z'))) {
      parts.emplace_back(text.substr(byte(start), byte(i) - byte(start)));
      start = j;
    }
    i = j;
  }
  parts.emplace_back(text.substr(byte(start), byte(e) - byte(start)));
  std::vector<std::string> merged;
  for (auto& p : parts) {
    if (!merged.empty() && count_words(merged.back()) < min_words) merged.back() += " " + p;
    else merged.push_back(std::move(p));
  }
  if (merged.size() > 1 && count_words(merged.back()) < min_words) {
    merged[merged.size() - 2] += " " + merged.back();
    merged.pop_back();
  }
  std::vector<std::string> out;
  for (const auto& m : merged) cap_comma(m, cap, out);
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
