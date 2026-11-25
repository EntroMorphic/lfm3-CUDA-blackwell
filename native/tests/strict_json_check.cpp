// validation/ttft-rest-v2 stage 1, rule 1: lfm::http::strict_json and lfm::Json::parse (a view parsed in place, plain
// runs scanned 8 bytes at a time, the result moved) against the references frozen from main e41e36c
// (json_reference_v2.hpp), and lfm3_text::parse_messages with copy_media false against true.
//   strict_json / Json::parse: N random documents (argv[1], default 1,000,000), each also changed by one byte replaced,
//     inserted or deleted, or cut short; plus clip-sized bodies. The same tree (every value, key order, the integral
//     flag) or the same refusal (exception type and message).
//   parse_messages: M random message lists (argv[2], default 100,000) with text, audio and image parts, valid and not:
//     the same messages (source() equal to the copy's data; with copy_media false, media parts hold no copy) or the
//     same refusal.
// One JSON line: {"ok":..., "documents":N, "accepted":A, "refused":R, "messages":M, "messages_accepted":..., ...}.
#include "lfm/lfm3_http.hpp"
#include "lfm/lfm3_text.hpp"
#include "lfm/model_config.hpp"

#include "json_reference_v2.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

namespace {
template <class J>
std::string show(const J& j) {  // canonical dump, exact for strings and doubles, with the integral flag
  std::string o;
  std::visit([&](const auto& v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, std::nullptr_t>) o = "null";
    else if constexpr (std::is_same_v<T, bool>) o = v ? "true" : "false";
    else if constexpr (std::is_same_v<T, double>) { char b[64]; std::snprintf(b, 64, "%a/%d", v, (int)j.integral); o = b; }
    else if constexpr (std::is_same_v<T, std::string>) { o = "s" + std::to_string(v.size()) + ":" + v; }
    else if constexpr (std::is_same_v<T, typename J::Array>) { o = "["; for (auto& e : v) o += show(e) + ","; o += "]"; }
    else { o = "{"; for (auto& [k, e] : v) o += "s" + std::to_string(k.size()) + ":" + k + "=" + show(e) + ","; o += "}"; }
  }, j.value);
  return o;
}
template <class F>
std::string outcome(F&& f) {
  try {
    return "OK " + show(f());
  } catch (const std::exception& e) {
    return std::string("ERR ") + typeid(e).name() + " " + e.what();
  }
}

std::mt19937_64 rng(20261123);
size_t pick(size_t n) { return size_t(rng() % n); }

const std::vector<std::string> kAtoms = {
    "\\\"", "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t", "\\u0041", "\\u00e9", "\\u20AC", "\\ud83d\\ude00",
    "\\ud83d", "\\ude00", "\\ud83dx", "\\ud83d\\u0041", "\\u12", "\\x", "\\", "\"", "\x01", "\x1f", "\x7f", "\t",
    "\xc3\xa9", "\xe2\x82\xac", "\xf0\x9f\x98\x80", "\xff", "\xc3", "\xed\xa0\x80", "\x80"};
const std::string kPlain = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/= .,:{}[]";

// A string's inside: plain runs of every length (so specials fall at every 8-byte alignment) and atoms
std::string inside() {
  std::string s;
  for (int k = int(pick(5)); k >= 0; --k) {
    for (size_t n = pick(41); n > 0; --n) s += kPlain[pick(kPlain.size())];
    if (pick(3)) s += kAtoms[pick(kAtoms.size())];
  }
  return s;
}
std::string space() {
  static const char* ws[] = {"", "", "", " ", "\n", "\t", "\r\n  "};
  return ws[pick(7)];
}
std::string value(int depth) {
  static const std::vector<std::string> numbers = {"0", "-0", "1", "-1", "42", "1.5", "-2.25e-3", "1e308", "1e309",
                                                   "-1e400", "1e-400", "2.5E+3", "00", "01", "1.", "-", ".5", "1e",
                                                   "123456789012345678901234567890", "3.14159", "9007199254740993"};
  static const std::vector<std::string> literals = {"true", "false", "null", "tru", "nul", "falsey", "NaN", "Infinity"};
  const size_t kind = depth > 6 ? pick(3) : pick(6);
  switch (kind) {
    case 0: return "\"" + inside() + "\"";
    case 1: return numbers[pick(numbers.size())];
    case 2: return literals[pick(literals.size())];
    case 3: {
      std::string s = "[" + space();
      for (size_t n = pick(5), i = 0; i < n; ++i) s += (i ? "," + space() : "") + value(depth + 1) + space();
      return s + "]";
    }
    default: {
      std::string s = "{" + space();
      std::vector<std::string> keys;
      for (size_t n = pick(5), i = 0; i < n; ++i) {
        std::string key = keys.empty() || pick(8) ? inside() : keys[pick(keys.size())];  // sometimes a duplicate key
        keys.push_back(key);
        s += (i ? "," + space() : "") + "\"" + key + "\"" + space() + ":" + space() + value(depth + 1) + space();
      }
      return s + "}";
    }
  }
}
std::string document() {
  std::string d;
  const size_t top = pick(20);
  if (top == 0) {  // nesting around the limit (64)
    const int n = 58 + int(pick(13));
    std::string closers;
    for (int i = 0; i < n; ++i) {
      const bool array = pick(2);
      d += array ? "[" : "{\"k\":";
      closers.insert(closers.begin(), array ? ']' : '}');
    }
    return d + value(7) + closers;
  }
  if (top == 1) return value(0);  // any value at the top (strict_json refuses non-objects)
  d = "{" + space();
  for (size_t n = 1 + pick(4), i = 0; i < n; ++i)
    d += (i ? "," + space() : "") + "\"" + inside() + "\":" + space() + value(1) + space();
  return space() + d + "}" + space();
}
std::string mutate(std::string d) {
  const std::string specials = std::string("\"\\{}[]:, \x01\x80\xff") + "0e.-";
  switch (pick(6)) {
    case 0: case 1: break;
    case 2: if (!d.empty()) d[pick(d.size())] = pick(2) ? specials[pick(specials.size())] : char(rng() & 255); break;
    case 3: d.insert(pick(d.size() + 1), 1, pick(2) ? specials[pick(specials.size())] : char(rng() & 255)); break;
    case 4: if (!d.empty()) d.erase(pick(d.size()), 1); break;
    case 5: d.resize(pick(d.size() + 1)); break;
  }
  return d;
}

long differ = 0;
void report(const char* what, long i, const std::string& doc, const std::string& a, const std::string& b) {
  if (++differ <= 5)
    std::fprintf(stderr, "%s differs on document %ld (%zu bytes): new %.120s | reference %.120s\n", what, i, doc.size(),
                 a.c_str(), b.c_str());
}

// parse_messages: copy_media false against true
std::string messages_outcome(const lfm::Json& j, bool audio, bool image, bool copy) {
  try {
    const auto ms = lfm::lfm3_text::parse_messages(j, audio, image, copy);
    std::string o = "OK";
    for (const auto& m : ms) {
      o += "|" + m.role + "/" + std::to_string(m.content_is_list) + "/" + m.content + "/" + std::to_string(m.has_thinking) +
           "/" + m.thinking + "/" + std::to_string(m.has_tool_call_id) + "/" + m.tool_call_id;
      for (const auto& p : m.parts) {
        if (!copy && (p.audio || p.image) && (!p.data.empty() || !p.media)) return "COPIED A MEDIA PART";
        o += "<" + p.text + "/" + std::to_string(p.audio) + "/" + std::to_string(p.image) + "/" + p.source() + ">";
      }
    }
    return o;
  } catch (const std::exception& e) {
    return std::string("ERR ") + typeid(e).name() + " " + e.what();
  }
}
std::string message_list() {
  static const char* roles[] = {"system", "user", "assistant", "tool", "bot"};
  std::string s = "[";
  for (size_t n = pick(5), i = 0; i < n; ++i) {
    std::string m = std::string("{\"role\":\"") + roles[pick(5)] + "\",\"content\":";
    if (pick(3) == 0) {
      m += "\"" + inside() + "\"";
    } else {
      m += "[";
      for (size_t k = pick(4), j = 0; j < k; ++j) {
        std::string part;
        switch (pick(6)) {
          case 0: part = "{\"type\":\"text\",\"text\":\"" + inside() + "\"}"; break;
          case 1: part = "{\"type\":\"audio\",\"audio\":\"" + (pick(6) ? std::string(pick(300), 'Q') : "") + "\"}"; break;
          case 2: part = "{\"type\":\"image\",\"image\":\"" + std::string(pick(300), 'R') + "\"}"; break;
          case 3: part = "{\"type\":\"audio\",\"audio\":7}"; break;
          case 4: part = "{\"type\":\"image\",\"image\":\"x\",\"extra\":1}"; break;
          default: part = "{\"type\":\"text\",\"text\":\"<|audio|>\"}"; break;
        }
        m += (j ? "," : "") + part;
      }
      m += "]";
    }
    if (pick(6) == 0) m += ",\"thinking\":\"" + inside() + "\"";
    if (pick(8) == 0) m += ",\"tool_call_id\":\"" + inside() + "\"";
    s += (i ? "," : "") + m + "}";
  }
  return s + "]";
}
}  // namespace

int main(int argc, char** argv) try {
  const long documents = argc > 1 ? std::stol(argv[1]) : 1000000;
  const long lists = argc > 2 ? std::stol(argv[2]) : 100000;
  long accepted = 0, refused = 0;
  auto check = [&](long i, const std::string& d) {
    const auto a = outcome([&] { return lfm::http::strict_json(d); });
    const auto b = outcome([&] { return lfm_v2::strict_json(d); });
    if (a != b) report("strict_json", i, d, a, b);
    (a.rfind("OK", 0) == 0 ? accepted : refused)++;
    const auto c = outcome([&] { return lfm::Json::parse(d); });
    const auto e = outcome([&] { return lfm_v2::Json::parse(d); });
    if (c != e) report("Json::parse", i, d, c, e);
  };
  for (long i = 0; i < documents; ++i) check(i, mutate(document()));
  // clip-sized bodies: a 90 s clip's base64 (3.84 MB) and others, plain and with one special at a random place
  for (const size_t n : {size_t(3840000), size_t(1000003), size_t(65537), size_t(4097)}) {
    std::string b64(n, 'A');
    for (auto& c : b64) c = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[pick(64)];
    const std::string head = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"audio\", \"audio\": \"";
    const std::string tail = "\"}]}], \"mode\": \"text\", \"max_new_tokens\": 32, \"seed\": 0, \"stream\": true}";
    check(-1, head + b64 + tail);
    for (const std::string& atom : kAtoms) {
      std::string s = b64;
      s.insert(pick(s.size()), atom);
      check(-1, head + s + tail);
    }
  }
  long messages_accepted = 0, messages_refused = 0;
  for (long i = 0; messages_accepted + messages_refused < lists; ++i) {
    const std::string text = message_list();
    lfm::Json j;
    try {
      j = lfm::Json::parse(text);
    } catch (const std::exception&) {
      continue;  // a list whose generated text is not JSON (an atom ended a string early): not a messages case
    }
    const bool audio = pick(4) != 0, image = pick(4) != 0;
    const auto a = messages_outcome(j, audio, image, false), b = messages_outcome(j, audio, image, true);
    if (a != b) report("parse_messages", i, text, a, b);
    (a.rfind("OK", 0) == 0 ? messages_accepted : messages_refused)++;
  }
  const bool ok = differ == 0 && accepted > 1000 && refused > 1000 && messages_accepted > 1000 && messages_refused > 1000;
  std::printf("{\"ok\":%s,\"documents\":%ld,\"accepted\":%ld,\"refused\":%ld,\"message_lists\":%ld,"
              "\"messages_accepted\":%ld,\"messages_refused\":%ld,\"differ\":%ld}\n",
              ok ? "true" : "false", documents, accepted, refused, lists, messages_accepted, messages_refused, differ);
  return ok ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "strict-json-check: %s\n", e.what());
  return 2;
}
