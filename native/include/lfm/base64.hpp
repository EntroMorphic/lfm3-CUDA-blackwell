#pragma once
// Strict standard base64 (RFC 4648, "+/" alphabet, "=" padding, no whitespace) for audio input (#3): a WAV clip
// travels base64 in the /speech request and in the engine frame; streamed audio out travels the same way. Host-only.
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfm::base64 {

// Refuses a length that is not a multiple of 4, a character outside the alphabet, padding anywhere but the last one or
// two characters, and nonzero bits under the padding (so every byte string has exactly one accepted encoding).
// validation/ttft-rest-v1: by a table, four characters to three bytes into a buffer sized once (about 5 ns a character
// before, on Thor). The same function of its input as the character-by-character original: the same bytes, refused
// for exactly the same strings (tests/test_audio_input_host.py compares the two on a million strings).
namespace detail {
struct DecodeTable {
  signed char v[256];
  constexpr DecodeTable() : v{} {
    for (int i = 0; i < 256; ++i) v[i] = -1;
    for (int i = 0; i < 26; ++i) v['A' + i] = static_cast<signed char>(i), v['a' + i] = static_cast<signed char>(26 + i);
    for (int i = 0; i < 10; ++i) v['0' + i] = static_cast<signed char>(52 + i);
    v[static_cast<unsigned char>('+')] = 62;
    v[static_cast<unsigned char>('/')] = 63;
  }
};
inline constexpr DecodeTable kDecode{};
}  // namespace detail

inline std::string decode(std::string_view s) {
  auto fail = []() -> std::string { throw std::invalid_argument("invalid base64"); };
  if (s.size() % 4) return fail();
  const size_t groups = s.size() / 4;
  if (groups == 0) return std::string();
  const auto* u = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char* last = u + 4 * (groups - 1);
  // padding: only in the last group, "x===" never, "xx==" or "xxx="
  const int pad = (last[3] == '=') + (last[2] == '=' && last[3] == '=');
  std::string out(3 * groups - size_t(pad), '\0');
  char* o = out.data();
  const signed char* t = detail::kDecode.v;
  for (; u < last; u += 4, o += 3) {
    const int a = t[u[0]], b = t[u[1]], c = t[u[2]], d = t[u[3]];
    if ((a | b | c | d) < 0) return fail();
    const unsigned n = unsigned(a) << 18 | unsigned(b) << 12 | unsigned(c) << 6 | unsigned(d);
    o[0] = char(n >> 16), o[1] = char((n >> 8) & 255), o[2] = char(n & 255);
  }
  int v[4];
  for (int k = 0; k < 4; ++k) {
    if (k >= 4 - pad) { v[k] = 0; continue; }
    v[k] = t[last[k]];
    if (v[k] < 0) return fail();
  }
  const unsigned n = unsigned(v[0]) << 18 | unsigned(v[1]) << 12 | unsigned(v[2]) << 6 | unsigned(v[3]);
  o[0] = char(n >> 16);
  if (pad < 2) o[1] = char((n >> 8) & 255);
  else if (n & 0xFFFF) return fail();
  if (pad < 1) o[2] = char(n & 255);
  else if (n & 0xFF) return fail();
  return out;
}

inline std::string encode(std::string_view bytes) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const size_t left = bytes.size() - i;
    unsigned n = unsigned(static_cast<unsigned char>(bytes[i])) << 16;
    if (left > 1) n |= unsigned(static_cast<unsigned char>(bytes[i + 1])) << 8;
    if (left > 2) n |= unsigned(static_cast<unsigned char>(bytes[i + 2]));
    out += t[n >> 18];
    out += t[(n >> 12) & 63];
    out += left > 1 ? t[(n >> 6) & 63] : '=';
    out += left > 2 ? t[n & 63] : '=';
  }
  return out;
}

}  // namespace lfm::base64
