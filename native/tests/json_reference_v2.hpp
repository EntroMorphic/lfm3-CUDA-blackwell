// validation/ttft-rest-v2 stage 1: lfm::Json (native/include/lfm/model_config.hpp) and lfm::http::strict_json
// (native/include/lfm/lfm3_http.hpp) as of main e41e36c (2026-11-23), frozen as the references for
// strict_json_check.cpp: the parser that reads a view in place, scans plain runs 8 bytes at a time and returns by move
// must give the same values and the same errors. Copied verbatim; namespace lfm_v2 only.
#pragma once
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "lfm/lfm3_text.hpp"

namespace lfm_v2 {

class Json {
 public:
  using Object = std::map<std::string, Json>;
  using Array = std::vector<Json>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
  // True for a number written without a fraction or exponent (Python json's int); numbers stay doubles.
  bool integral = false;
  const Json& at(const std::string& key) const {
    const auto* o = std::get_if<Object>(&value);
    if (!o || !o->count(key)) throw std::runtime_error("missing configuration field: " + key);
    return o->at(key);
  }
  bool has(const std::string& key) const {
    const auto* o = std::get_if<Object>(&value);
    return o && o->count(key);
  }
  double number() const {
    if (const auto* n = std::get_if<double>(&value)) return *n;
    throw std::runtime_error("configuration field must be a number");
  }
  int integer() const {
    const double n = number();
    if (std::floor(n) != n || n < 0 || n > std::numeric_limits<int>::max())
      throw std::runtime_error("configuration integer out of range");
    return static_cast<int>(n);
  }
  std::string string() const {
    if (const auto* s = std::get_if<std::string>(&value)) return *s;
    throw std::runtime_error("configuration field must be a string");
  }
  bool boolean() const {
    if (const auto* b = std::get_if<bool>(&value)) return *b;
    throw std::runtime_error("configuration field must be boolean");
  }
  const Array& array() const {
    if (const auto* a = std::get_if<Array>(&value)) return *a;
    throw std::runtime_error("configuration field must be an array");
  }
  static Json parse(const std::string& text) {
    size_t p = 0;
    const Json result = parse_value(text, p, 0);
    space(text, p);
    if (p != text.size()) throw std::runtime_error("trailing configuration JSON");
    return result;
  }
 private:
  static void space(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
  }
  static char take(const std::string& s, size_t& p) {
    if (p == s.size()) throw std::runtime_error("truncated configuration JSON");
    return s[p++];
  }
  static unsigned hex4(const std::string& s, size_t& p) {
    unsigned n = 0;
    for (int i = 0; i < 4; ++i) {
      const char h = take(s, p);
      const int d = h >= '0' && h <= '9' ? h - '0' :
                    h >= 'a' && h <= 'f' ? h - 'a' + 10 :
                    h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
      if (d < 0) throw std::runtime_error("invalid JSON Unicode escape");
      n = n * 16 + d;
    }
    return n;
  }
  static void utf8(std::string& out, unsigned n) {
    if (n < 128) out += static_cast<char>(n);
    else if (n < 2048) { out += static_cast<char>(0xc0 | (n >> 6)); out += static_cast<char>(0x80 | (n & 63)); }
    else if (n < 65536) {
      out += static_cast<char>(0xe0 | (n >> 12)); out += static_cast<char>(0x80 | ((n >> 6) & 63)); out += static_cast<char>(0x80 | (n & 63));
    } else {
      out += static_cast<char>(0xf0 | (n >> 18)); out += static_cast<char>(0x80 | ((n >> 12) & 63));
      out += static_cast<char>(0x80 | ((n >> 6) & 63)); out += static_cast<char>(0x80 | (n & 63));
    }
  }
  static std::string quoted(const std::string& s, size_t& p) {
    if (take(s, p) != '"') throw std::runtime_error("expected JSON string");
    std::string out;
    for (;;) {
      // the plain run up to the next quote, backslash or control character, appended at once (the same bytes as one
      // at a time: a 13 MB base64 image part parsed ~6x faster)
      size_t run = p;
      while (run < s.size()) {
        const unsigned char u = static_cast<unsigned char>(s[run]);
        if (u == '"' || u == '\\' || u < 32) break;
        ++run;
      }
      out.append(s, p, run - p);
      p = run;
      char c = take(s, p);
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 32) throw std::runtime_error("control character in JSON string");
      if (c == '\\') {
        c = take(s, p);
        switch (c) {
          case '"': case '\\': case '/': break;
          case 'b': c = '\b'; break; case 'f': c = '\f'; break;
          case 'n': c = '\n'; break; case 'r': c = '\r'; break; case 't': c = '\t'; break;
          case 'u': {
            unsigned n = hex4(s, p);
            if (n >= 0xd800 && n <= 0xdbff) {
              if (take(s, p) != '\\' || take(s, p) != 'u') throw std::runtime_error("missing JSON low surrogate");
              const unsigned low = hex4(s, p);
              if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("invalid JSON low surrogate");
              n = 0x10000 + ((n - 0xd800) << 10) + low - 0xdc00;
            } else if (n >= 0xdc00 && n <= 0xdfff) {
              throw std::runtime_error("unpaired JSON low surrogate");
            }
            utf8(out, n);
            continue;
          }
          default: throw std::runtime_error("invalid JSON escape");
        }
      }
      out += c;
    }
  }
  static Json parse_value(const std::string& s, size_t& p, int depth) {
    if (depth > 64) throw std::runtime_error("configuration JSON nesting too deep");
    space(s, p);
    if (p == s.size()) throw std::runtime_error("missing configuration JSON value");
    const char c = s[p];
    if (c == '"') return Json{quoted(s, p)};
    if (c == '{' || c == '[') {
      ++p;
      Object object;
      Array array;
      space(s, p);
      const char end = c == '{' ? '}' : ']';
      if (p < s.size() && s[p] == end) { ++p; return c == '{' ? Json{std::move(object)} : Json{std::move(array)}; }
      for (;;) {
        space(s, p);
        if (c == '{') {
          const auto key = quoted(s, p);
          space(s, p);
          if (take(s, p) != ':') throw std::runtime_error("expected JSON colon");
          if (!object.emplace(key, parse_value(s, p, depth + 1)).second)
            throw std::runtime_error("duplicate configuration key: " + key);
        } else array.push_back(parse_value(s, p, depth + 1));
        space(s, p);
        const char separator = take(s, p);
        if (separator == end) return c == '{' ? Json{std::move(object)} : Json{std::move(array)};  // moved: a copy
        // here copied every nested value once per level
        if (separator != ',') throw std::runtime_error("invalid JSON separator");
      }
    }
    for (const auto& literal : {std::string("true"), std::string("false"), std::string("null")}) {
      if (s.compare(p, literal.size(), literal) == 0) {
        p += literal.size();
        if (literal == "null") return Json{nullptr};
        return Json{literal == "true"};
      }
    }
    const size_t begin = p;
    if (s[p] == '-') ++p;
    if (p == s.size() || s[p] < '0' || s[p] > '9') throw std::runtime_error("invalid JSON number");
    if (s[p] == '0') ++p;
    else while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
    bool integral = true;
    if (p < s.size() && s[p] == '.') {
      integral = false;
      const size_t digits = ++p;
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
      if (p == digits) throw std::runtime_error("invalid JSON fraction");
    }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
      integral = false;
      ++p;
      if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
      const size_t digits = p;
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
      if (p == digits) throw std::runtime_error("invalid JSON exponent");
    }
    double n = 0;
    try {
      n = std::stod(s.substr(begin, p - begin));
    } catch (const std::out_of_range&) {  // overflow, or underflow to a subnormal/zero (ERANGE)
      throw std::runtime_error("JSON number outside the double range");
    }
    if (!std::isfinite(n)) throw std::runtime_error("nonfinite JSON number");
    Json number{n};
    number.integral = integral;
    return number;
  }
};

// lfm3_worker_client.strict_json: strict UTF-8, nesting at most 64, no duplicate keys, no NaN/Infinity,
// finite numbers, an object at the top.
inline Json strict_json(std::string_view raw) {
  if (!lfm::lfm3_text::valid_utf8(raw)) throw std::invalid_argument("invalid UTF-8");
  bool quoted = false, escaped = false;
  int depth = 0;
  for (const char c : raw) {
    if (quoted) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') quoted = false;
    } else if (c == '"') quoted = true;
    else if (c == '[' || c == '{') { if (++depth > 64) throw std::invalid_argument("JSON nesting exceeds limit"); }
    else if (c == ']' || c == '}') --depth;
  }
  Json j = Json::parse(std::string(raw));
  if (!std::holds_alternative<Json::Object>(j.value)) throw std::invalid_argument("expected JSON object");
  return j;
}

}  // namespace lfm_v2
