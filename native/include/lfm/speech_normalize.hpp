#pragma once
// Text normalization for speech (validation/speech-normalize-v1, which chose these rules by test): numbers, symbols and
// abbreviations written out as words before the sentence split, American English. A native port of
// tools/speech_normalize.py, the reference: the same passes in the same order, each a hand-written scanner that finds
// the same matches as the reference's regular expression (leftmost, non-overlapping, with the same backtracking
// outcome), so the output is byte for byte the reference's (tests/test_speech_normalize.py). No std::regex: every pass
// is linear in the text apart from bounded look-arounds. Matching is ASCII (the reference's (?a)): digits, word
// characters, spaces and word boundaries are ASCII; the non-ASCII symbols (− £ € ° × – “) are matched as UTF-8
// sequences; the few places where the reference counts code points (context windows, Python's str.rstrip) count code
// points here too. Host-only.
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfm::speech_normalize {

namespace detail {

inline bool digit(char c) { return c >= '0' && c <= '9'; }
inline bool upper(char c) { return c >= 'A' && c <= 'Z'; }
inline bool lower(char c) { return c >= 'a' && c <= 'z'; }
inline bool alpha(char c) { return upper(c) || lower(c); }
inline bool word(char c) { return alpha(c) || digit(c) || c == '_'; }
inline bool space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// The text one pass reads, with the reference's character tests at byte positions (out of range: no character).
struct Text {
  std::string_view s;
  size_t n() const { return s.size(); }
  char at(size_t i) const { return i < s.size() ? s[i] : '\0'; }
  bool is_digit(size_t i) const { return i < s.size() && digit(s[i]); }
  bool is_word(size_t i) const { return i < s.size() && word(s[i]); }
  bool is_space(size_t i) const { return i < s.size() && space(s[i]); }
  bool prev_word(size_t i) const { return i > 0 && word(s[i - 1]); }
  bool boundary(size_t i) const { return prev_word(i) != is_word(i); }  // \b
  bool starts(size_t i, std::string_view x) const { return i <= s.size() && s.substr(i).substr(0, x.size()) == x; }
  size_t digits_end(size_t i) const { while (is_digit(i)) ++i; return i; }
  size_t spaces_end(size_t i) const { while (is_space(i)) ++i; return i; }
};

// A pass: at each position, the matcher returns the match's end and fills the replacement, or 0 for no match there.
template <class Matcher>
std::string pass(const std::string& in, Matcher&& match) {
  const Text t{in};
  std::string out;
  out.reserve(in.size() + in.size() / 4);
  std::string rep;
  for (size_t i = 0; i < in.size();) {
    rep.clear();
    const size_t end = match(t, i, rep);
    if (end > i) { out += rep; i = end; }
    else out += in[i++];
  }
  return out;
}

inline const char* kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                              "nineteen"};
inline const char* kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
inline const char* kMonths[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
                                "October", "November", "December"};

// Numbers as decimal digit strings, of any length (the reference uses Python's integers).
inline std::string strip_zeros(std::string_view d) {
  size_t k = 0;
  while (k + 1 < d.size() && d[k] == '0') ++k;
  return std::string(d.substr(k));
}
inline int small(std::string_view d) {  // the value of a digit string known to be short (< 10 digits)
  int v = 0;
  for (char c : d) v = v * 10 + (c - '0');
  return v;
}
inline bool is_zero(std::string_view d) { return strip_zeros(d) == "0"; }
inline bool equals(std::string_view d, long v) { const auto s = strip_zeros(d); return s.size() < 10 && small(s) == v; }
inline bool in_range(std::string_view d, long lo, long hi) {
  const auto s = strip_zeros(d);
  if (s.size() >= 10) return false;
  const long v = small(s);
  return lo <= v && v <= hi;
}

inline std::string cardinal_small(int n) {  // n < 1000
  if (n < 20) return kOnes[n];
  if (n < 100) return std::string(kTens[n / 10]) + (n % 10 ? std::string("-") + kOnes[n % 10] : "");
  return std::string(kOnes[n / 100]) + " hundred" + (n % 100 ? " " + cardinal_small(n % 100) : "");
}
// cardinal(int(d)): scales trillion, billion, million, thousand, largest first, recursively (as the reference)
inline std::string cardinal(std::string_view digits_in) {
  const std::string d = strip_zeros(digits_in);
  if (d.size() <= 3) return cardinal_small(small(d));
  static const std::pair<size_t, const char*> scales[] = {{12, "trillion"}, {9, "billion"}, {6, "million"}, {3, "thousand"}};
  for (const auto& [k, name] : scales)
    if (d.size() > k) {
      const std::string hi = d.substr(0, d.size() - k), lo = strip_zeros(d.substr(d.size() - k));
      return cardinal(hi) + " " + name + (lo != "0" ? " " + cardinal(lo) : "");
    }
  return {};  // unreachable
}
inline std::string cardinal(long n) { return cardinal(std::to_string(n)); }

inline std::string ordinal(std::string_view d) {
  const std::string w = cardinal(d);
  const size_t last_space = w.rfind(' ');
  const std::string last_word = last_space == std::string::npos ? w : w.substr(last_space + 1);
  const char sep = last_word.find('-') != std::string::npos ? '-' : ' ';
  const size_t at = w.rfind(sep);
  const std::string head = at == std::string::npos ? "" : w.substr(0, at + 1);
  std::string last = at == std::string::npos ? w : w.substr(at + 1);
  static const std::pair<const char*, const char*> irregular[] = {
      {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"}, {"eight", "eighth"},
      {"nine", "ninth"}, {"twelve", "twelfth"}};
  bool done = false;
  for (const auto& [a, b] : irregular)
    if (last == a) { last = b; done = true; break; }
  if (!done) last = !last.empty() && last.back() == 'y' ? last.substr(0, last.size() - 1) + "ieth" : last + "th";
  return head + last;
}
inline std::string ordinal(long n) { return ordinal(std::to_string(n)); }

// pairs(n): years, decades and house numbers (n < 100000)
inline std::string pairs(long n) {
  if (n < 100 || (n % 1000 == 0 && n < 10000) || (2000 <= n && n < 2010)) return cardinal(n);
  if (n < 1000)
    return n % 100 == 0 ? cardinal(n)
                        : std::string(kOnes[n / 100]) + " " +
                              (n % 100 >= 10 ? cardinal(n % 100) : std::string("oh ") + kOnes[n % 100]);
  const long hi = n / 100, lo = n % 100;
  if (lo == 0) return cardinal(hi) + " hundred";
  return cardinal(hi) + " " + (lo >= 10 ? cardinal(lo) : std::string("oh ") + kOnes[lo]);
}
inline std::string plural_pairs(const std::string& w) {
  const size_t at = w.rfind(' ');
  const std::string head = at == std::string::npos ? "" : w.substr(0, at + 1);
  const std::string last = at == std::string::npos ? w : w.substr(at + 1);
  return head + (!last.empty() && last.back() == 'y' ? last.substr(0, last.size() - 1) + "ies" : last + "s");
}
inline std::string spell_digits(std::string_view d, const char* zero = "zero") {
  std::string out;
  for (char c : d) {
    if (!out.empty()) out += ' ';
    out += c == '0' ? zero : kOnes[c - '0'];
  }
  return out;
}
inline std::string without_commas(std::string_view s) {
  std::string out;
  for (char c : s) if (c != ',') out += c;
  return out;
}
// number(s): commas dropped; "whole.frac" as "<whole> point <digits>"
inline std::string number(std::string_view raw) {
  const std::string s = without_commas(raw);
  const size_t dot = s.find('.');
  if (dot == std::string::npos) return cardinal(s);
  const std::string whole = s.substr(0, dot);
  return cardinal(whole.empty() ? std::string("0") : whole) + " point " + spell_digits(std::string_view(s).substr(dot + 1));
}

// "\s*$|\s+[A-Z...]" at i: only whitespace remains, or whitespace then a capital (or, with quotes, a quote or bracket)
inline bool sentence_end(const Text& t, size_t i, bool quotes) {
  const size_t j = t.spaces_end(i);
  if (j == t.n()) return true;
  if (j == i) return false;
  if (upper(t.at(j))) return true;
  return quotes && (t.at(j) == '"' || t.at(j) == '\'' || t.at(j) == '(' || t.starts(j, "\xE2\x80\x9C"));
}
inline bool cont(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }
// the byte offset k code points before `end` (0 if fewer)
inline size_t back_code_points(std::string_view s, size_t end, size_t k) {
  size_t i = end;
  while (k > 0 && i > 0) {
    --i;
    while (i > 0 && cont(s[i])) --i;
    --k;
  }
  return i;
}
// Python's str.isspace for the code point ending at `end`; returns its start, or npos when not a space
inline size_t python_space_before(std::string_view s, size_t end) {
  if (end == 0) return std::string::npos;
  size_t i = end - 1;
  while (i > 0 && cont(s[i])) --i;
  const auto* p = reinterpret_cast<const unsigned char*>(s.data() + i);
  const size_t len = end - i;
  uint32_t cp;
  if (len == 1 && p[0] < 0x80) cp = p[0];
  else if (len == 2 && (p[0] & 0xE0) == 0xC0) cp = (p[0] & 0x1Fu) << 6 | (p[1] & 0x3Fu);
  else if (len == 3 && (p[0] & 0xF0) == 0xE0) cp = (p[0] & 0x0Fu) << 12 | (p[1] & 0x3Fu) << 6 | (p[2] & 0x3Fu);
  else return std::string::npos;  // 4-byte code points are never spaces; invalid UTF-8 is not either
  const bool sp = (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
                  cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
                  cp == 0x205F || cp == 0x3000;
  return sp ? i : std::string::npos;
}
inline size_t python_rstrip(std::string_view s, size_t end) {
  for (size_t at; (at = python_space_before(s, end)) != std::string::npos;) end = at;
  return end;
}
inline std::vector<std::string> letter_runs(std::string_view s) {  // re.findall("[A-Za-z]+")
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    if (!alpha(s[i])) { ++i; continue; }
    size_t j = i;
    while (j < s.size() && alpha(s[j])) ++j;
    out.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}
inline std::string ascii_lower(std::string s) {
  for (char& c : s) if (upper(c)) c = char(c - 'A' + 'a');
  return s;
}
// an optional minus sign ("-" or U+2212) at i: its length
inline size_t minus_sign(const Text& t, size_t i) {
  if (t.at(i) == '-') return 1;
  return t.starts(i, "\xE2\x88\x92") ? 3 : 0;
}
// \d[\d,]*(?:\.\d+)? at i (the first character a digit): its end
inline size_t amount_end(const Text& t, size_t i) {
  if (!t.is_digit(i)) return i;
  size_t j = i + 1;
  while (t.is_digit(j) || t.at(j) == ',') ++j;
  if (t.at(j) == '.' && t.is_digit(j + 1)) j = t.digits_end(j + 1);
  return j;
}
// \d+(?:\.\d+)? at i: its end
inline size_t decimal_end(const Text& t, size_t i) {
  if (!t.is_digit(i)) return i;
  size_t j = t.digits_end(i);
  if (t.at(j) == '.' && t.is_digit(j + 1)) j = t.digits_end(j + 1);
  return j;
}
inline bool not_after(const Text& t, size_t i, std::string_view chars, bool or_word) {  // (?<![\w<chars>])
  if (i == 0) return true;
  const char c = t.s[i - 1];
  return !(or_word && word(c)) && chars.find(c) == std::string_view::npos;
}

}  // namespace detail

inline std::string normalize(const std::string& text) {
  using namespace detail;
  std::string t = text;

  // phone numbers: \(?\b(\d{3})\)?[-. ](\d{3})-(\d{4})\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    size_t j = i + (x.at(i) == '(' ? 1 : 0);
    if (!x.boundary(j) || x.digits_end(j) < j + 3) return 0;
    const size_t a = j;
    j += 3;
    if (x.at(j) == ')') ++j;
    if (x.at(j) != '-' && x.at(j) != '.' && x.at(j) != ' ') return 0;
    const size_t b = ++j;
    if (x.digits_end(j) < j + 3 || x.at(j + 3) != '-') return 0;
    const size_t c = j + 4;
    if (x.digits_end(c) < c + 4 || !x.boundary(c + 4)) return 0;
    r = spell_digits(x.s.substr(a, 3), "oh") + ", " + spell_digits(x.s.substr(b, 3), "oh") + ", " +
        spell_digits(x.s.substr(c, 4), "oh");
    return c + 4;
  });
  // (?<![\d.,-])\b(\d{3})-(\d{4})\b(?![-\d])
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, "0123456789.,-", false) || !x.boundary(i)) return 0;
    if (x.digits_end(i) < i + 3 || x.at(i + 3) != '-' || x.digits_end(i + 4) < i + 8) return 0;
    const size_t e = i + 8;
    if (!x.boundary(e) || x.at(e) == '-' || x.is_digit(e)) return 0;
    r = spell_digits(x.s.substr(i, 3), "oh") + ", " + spell_digits(x.s.substr(i + 4, 4), "oh");
    return e;
  });

  // money: ([$£€])(\d[\d,]*(?:\.\d+)?)(\s+(million|billion|trillion|thousand)\b)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const char* const units[][4] = {{"dollar", "dollars", "cent", "cents"}, {"pound", "pounds", "penny", "pence"},
                                           {"euro", "euros", "cent", "cents"}};
    int cur;
    size_t j;
    if (x.at(i) == '$') cur = 0, j = i + 1;
    else if (x.starts(i, "\xC2\xA3")) cur = 1, j = i + 2;
    else if (x.starts(i, "\xE2\x82\xAC")) cur = 2, j = i + 3;
    else return 0;
    const size_t e = amount_end(x, j);
    if (e == j) return 0;
    const std::string value = without_commas(x.s.substr(j, e - j));
    size_t end = e;
    const size_t k = x.spaces_end(e);
    if (k > e)
      for (const char* scale : {"million", "billion", "trillion", "thousand"})
        if (x.starts(k, scale) && x.boundary(k + std::string_view(scale).size())) {
          end = k + std::string_view(scale).size();
          r = number(value) + " " + scale + " " + units[cur][1];
          return end;
        }
    const size_t dot = value.find('.');
    const std::string whole = value.substr(0, dot);
    const std::string frac = dot == std::string::npos ? "" : value.substr(dot + 1);
    const int cents = frac.empty() ? 0 : small((frac + "00").substr(0, 2));
    if (!is_zero(whole) || !cents) r = cardinal(whole) + " " + (equals(whole, 1) ? units[cur][0] : units[cur][1]);
    if (cents) r += (r.empty() ? "" : " and ") + cardinal(long(cents)) + " " + (cents == 1 ? units[cur][2] : units[cur][3]);
    return end;
  });
  // numeric dates: \b(1[0-2]|0?[1-9])/(3[01]|[12]\d|0?[1-9])/(\d{4})\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    auto alternatives = [&](size_t at, bool day) {  // the lengths the alternation can take at `at`, in its order
      std::vector<size_t> out;
      if (!day) {
        if (x.at(at) == '1' && x.at(at + 1) >= '0' && x.at(at + 1) <= '2') out.push_back(2);
      } else {
        if (x.at(at) == '3' && (x.at(at + 1) == '0' || x.at(at + 1) == '1')) out.push_back(2);
        if ((x.at(at) == '1' || x.at(at) == '2') && x.is_digit(at + 1)) out.push_back(2);
      }
      if (x.at(at) == '0' && x.at(at + 1) >= '1' && x.at(at + 1) <= '9') out.push_back(2);
      if (x.at(at) >= '1' && x.at(at) <= '9') out.push_back(1);
      return out;
    };
    for (size_t m : alternatives(i, false)) {
      if (x.at(i + m) != '/') continue;
      const size_t d0 = i + m + 1;
      for (size_t d : alternatives(d0, true)) {
        const size_t y0 = d0 + d;
        if (x.at(y0) != '/' || x.digits_end(y0 + 1) < y0 + 5 || !x.boundary(y0 + 5)) continue;
        r = std::string(kMonths[small(x.s.substr(i, m)) - 1]) + " " + ordinal(x.s.substr(d0, d)) + ", " +
            pairs(small(x.s.substr(y0 + 1, 4)));
        return y0 + 5;
      }
    }
    return 0;
  });

  // times: \b([01]?\d|2[0-3]):([0-5]\d)\b(?=(\s*[aApP]\.?[mM]\b\.?)?)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    std::vector<size_t> hours;
    if ((x.at(i) == '0' || x.at(i) == '1') && x.is_digit(i + 1)) hours.push_back(2);
    if (x.is_digit(i)) hours.push_back(1);
    if (x.at(i) == '2' && x.at(i + 1) >= '0' && x.at(i + 1) <= '3') hours.push_back(2);
    for (size_t h : hours) {
      const size_t c = i + h;
      if (x.at(c) != ':' || !(x.at(c + 1) >= '0' && x.at(c + 1) <= '5') || !x.is_digit(c + 2) || !x.boundary(c + 3))
        continue;
      const size_t e = c + 3;
      bool meridiem = false;  // the look-ahead's optional group
      size_t j = x.spaces_end(e);
      const char a = x.at(j);
      if (a == 'a' || a == 'A' || a == 'p' || a == 'P') {
        ++j;
        size_t k = j;
        if (x.at(k) == '.' && (x.at(k + 1) == 'm' || x.at(k + 1) == 'M')) k += 2;
        else if (x.at(k) == 'm' || x.at(k) == 'M') k += 1;
        else k = 0;
        meridiem = k && x.boundary(k);
      }
      const int hh = small(x.s.substr(i, h)), mm = small(x.s.substr(c + 1, 2));
      if (mm == 0) r = cardinal(long(hh)) + (meridiem ? "" : " o'clock");
      else r = cardinal(long(hh)) + " " + (mm >= 10 ? cardinal(long(mm)) : std::string("oh ") + kOnes[mm]);
      return e;
    }
    return 0;
  });
  // month abbreviations before a number: \b(Jan|...|Dec)\.(?=\s+\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const std::pair<const char*, const char*> abbr[] = {
        {"Jan", "January"}, {"Feb", "February"}, {"Mar", "March"}, {"Apr", "April"}, {"Jun", "June"}, {"Jul", "July"},
        {"Aug", "August"}, {"Sep", "September"}, {"Sept", "September"}, {"Oct", "October"}, {"Nov", "November"},
        {"Dec", "December"}};
    if (!x.boundary(i)) return 0;
    for (const auto& [a, full] : abbr) {
      const size_t e = i + std::string_view(a).size();
      if (!x.starts(i, a) || x.at(e) != '.') continue;
      const size_t k = x.spaces_end(e + 1);
      if (k > e + 1 && x.is_digit(k)) { r = full; return e + 1; }
    }
    return 0;
  });
  // month-day dates: \b(January|...|December)\s+(3[01]|[12]\d|0?[1-9])(st|nd|rd|th)?\b(?![\d:])
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    for (const char* month : kMonths) {
      const size_t m = std::string_view(month).size();
      if (!x.starts(i, month)) continue;
      const size_t d0 = x.spaces_end(i + m);
      if (d0 == i + m) continue;
      std::vector<size_t> days;
      if (x.at(d0) == '3' && (x.at(d0 + 1) == '0' || x.at(d0 + 1) == '1')) days.push_back(2);
      if ((x.at(d0) == '1' || x.at(d0) == '2') && x.is_digit(d0 + 1)) days.push_back(2);
      if (x.at(d0) == '0' && x.at(d0 + 1) >= '1' && x.at(d0 + 1) <= '9') days.push_back(2);
      if (x.at(d0) >= '1' && x.at(d0) <= '9') days.push_back(1);
      for (size_t d : days) {
        const size_t s = d0 + d;
        std::vector<size_t> ends;
        for (const char* suffix : {"st", "nd", "rd", "th"})
          if (x.starts(s, suffix)) { ends.push_back(s + 2); break; }
        ends.push_back(s);
        for (size_t e : ends)
          if (x.boundary(e) && !x.is_digit(e) && x.at(e) != ':') {
            r = std::string(month) + " " + ordinal(x.s.substr(d0, d));
            return e;
          }
      }
      return 0;  // the months are distinct words: no other alternative can match here
    }
    return 0;
  });
  // temperatures: (?<![\w.])([-−])?(\d+(?:\.\d+)?)\s*°(?:\s*([CF])\b)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t sign = minus_sign(x, i), n0 = i + sign;
    const size_t ne = decimal_end(x, n0);
    if (ne == n0) return 0;
    const size_t deg = x.spaces_end(ne);
    if (!x.starts(deg, "\xC2\xB0")) return 0;
    size_t e = deg + 2;
    const char* unit = "";
    const size_t u = x.spaces_end(e);
    if ((x.at(u) == 'C' || x.at(u) == 'F') && x.boundary(u + 1)) {
      unit = x.at(u) == 'C' ? " Celsius" : " Fahrenheit";
      e = u + 1;
    }
    const std::string_view value = x.s.substr(n0, ne - n0);
    r = std::string(sign ? "minus " : "") + number(value) + " degree" + (value == "1" ? "" : "s") + unit;
    return e;
  });
  // percentages: (?<![\w.])([-−])?(\d[\d,]*(?:\.\d+)?)\s*%
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t sign = minus_sign(x, i), n0 = i + sign;
    const size_t ne = amount_end(x, n0);
    if (ne == n0) return 0;
    const size_t p = x.spaces_end(ne);
    if (x.at(p) != '%') return 0;
    r = std::string(sign ? "minus " : "") + number(x.s.substr(n0, ne - n0)) + " percent";
    return p + 1;
  });
  // addresses: \b(\d{1,5}) ((?:[A-Z][a-z]+ )+)(St|Street|...)\b(\.)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const std::pair<const char*, const char*> street[] = {
        {"St", "Street"}, {"Street", "Street"}, {"Ave", "Avenue"}, {"Avenue", "Avenue"}, {"Rd", "Road"},
        {"Road", "Road"}, {"Blvd", "Boulevard"}, {"Boulevard", "Boulevard"}, {"Dr", "Drive"}, {"Drive", "Drive"},
        {"Ln", "Lane"}, {"Lane", "Lane"}, {"Way", "Way"}, {"Ct", "Court"}, {"Court", "Court"}, {"Pl", "Place"},
        {"Place", "Place"}};
    if (!x.boundary(i)) return 0;
    const size_t de = x.digits_end(i);
    if (de == i || de - i > 5 || x.at(de) != ' ') return 0;
    std::vector<size_t> names;  // the end of each "Name " (after its space), greedy
    for (size_t j = de + 1; upper(x.at(j)) && lower(x.at(j + 1));) {
      size_t k = j + 1;
      while (lower(x.at(k))) ++k;
      if (x.at(k) != ' ') break;
      names.push_back(k + 1);
      j = k + 1;
    }
    for (size_t k = names.size(); k-- > 0;) {
      const size_t at = names[k];
      for (const auto& [key, full] : street) {
        const size_t e = at + std::string_view(key).size();
        if (!x.starts(at, key) || !x.boundary(e)) continue;
        const bool dot = x.at(e) == '.';
        const size_t end = e + (dot ? 1 : 0);
        r = pairs(small(x.s.substr(i, de - i))) + " " + std::string(x.s.substr(de + 1, at - de - 1)) + full +
            (dot && sentence_end(x, end, false) ? "." : "");
        return end;
      }
    }
    return 0;
  });
  // decades: \b(1\d|20)(\d0)s\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    const bool first = (x.at(i) == '1' && x.is_digit(i + 1)) || (x.at(i) == '2' && x.at(i + 1) == '0');
    if (!first || !x.is_digit(i + 2) || x.at(i + 3) != '0' || x.at(i + 4) != 's' || !x.boundary(i + 5)) return 0;
    r = plural_pairs(pairs(small(x.s.substr(i, 4))));
    return i + 5;
  });
  // numbers with units: (?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(units, longest first)\b(\.)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const char* const units[][3] = {
        {"mph", "mile per hour", "miles per hour"}, {"kph", "kilometer per hour", "kilometers per hour"},
        {"lbs", "pound", "pounds"}, {"kWh", "kilowatt hour", "kilowatt hours"}, {"hrs", "hour", "hours"},
        {"min", "minute", "minutes"}, {"sec", "second", "seconds"}, {"km", "kilometer", "kilometers"},
        {"kg", "kilogram", "kilograms"}, {"cm", "centimeter", "centimeters"}, {"mm", "millimeter", "millimeters"},
        {"mg", "milligram", "milligrams"}, {"ml", "milliliter", "milliliters"}, {"ft", "foot", "feet"},
        {"lb", "pound", "pounds"}, {"oz", "ounce", "ounces"}, {"mi", "mile", "miles"}, {"GB", "gigabyte", "gigabytes"},
        {"MB", "megabyte", "megabytes"}, {"hr", "hour", "hours"}};
    if (!not_after(x, i, ".", true)) return 0;
    const size_t ne = amount_end(x, i);
    if (ne == i) return 0;
    for (size_t at : {ne + (x.is_space(ne) ? 1 : 0), ne}) {
      for (const auto& u : units) {
        const size_t e = at + std::string_view(u[0]).size();
        if (!x.starts(at, u[0]) || !x.boundary(e)) continue;
        const std::string_view value = x.s.substr(i, ne - i);
        const bool dot = x.at(e) == '.';
        const size_t end = e + (dot ? 1 : 0);
        r = number(value) + " " + (value == "1" ? u[1] : u[2]) + (dot && sentence_end(x, end, true) ? "." : "");
        return end;
      }
      if (at == ne) break;
    }
    return 0;
  });
  // ordinals: \b(\d+)(st|nd|rd|th)\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    const size_t de = x.digits_end(i);
    if (de == i) return 0;
    for (const char* suffix : {"st", "nd", "rd", "th"})
      if (x.starts(de, suffix) && x.boundary(de + 2)) { r = ordinal(x.s.substr(i, de - i)); return de + 2; }
    return 0;
  });
  // fractions: (?<![\w/.])(\d+)/(\d+)\b(?!/)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, "/.", true)) return 0;
    const size_t a = x.digits_end(i);
    if (a == i || x.at(a) != '/') return 0;
    const size_t b = x.digits_end(a + 1);
    if (b == a + 1 || !x.boundary(b) || x.at(b) == '/') return 0;
    const std::string_view n = x.s.substr(i, a - i), d = x.s.substr(a + 1, b - a - 1);
    const bool one = equals(n, 1);
    if (equals(d, 2)) r = cardinal(n) + (one ? " half" : " halves");
    else if (equals(d, 4)) r = cardinal(n) + (one ? " quarter" : " quarters");
    else r = cardinal(n) + " " + ordinal(d) + (one ? "" : "s");
    return b;
  });
  // ranges: (?<![\w.,-])(\d+)\s?[-–]\s?(\d+)\b(?![-\d])
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,-", true)) return 0;
    const size_t a = x.digits_end(i);
    if (a == i) return 0;
    size_t j = a + (x.is_space(a) ? 1 : 0);
    if (x.at(j) == '-') j += 1;
    else if (x.starts(j, "\xE2\x80\x93")) j += 3;
    else return 0;
    j += x.is_space(j) ? 1 : 0;
    const size_t b = x.digits_end(j);
    if (b == j || !x.boundary(b) || x.at(b) == '-' || x.is_digit(b)) return 0;
    const std::string_view p = x.s.substr(i, a - i), q = x.s.substr(j, b - j);
    const bool years = in_range(p, 1000, 2099) && in_range(q, 1000, 2099);
    r = (years ? pairs(small(strip_zeros(p))) : cardinal(p)) + " to " + (years ? pairs(small(strip_zeros(q))) : cardinal(q));
    return b;
  });
  // "#12": #\s?(\d+)\b ; "No. 14": \bNo\.\s?(\d+)\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (x.at(i) != '#') return 0;
    const size_t j = i + 1 + (x.is_space(i + 1) ? 1 : 0), e = x.digits_end(j);
    if (e == j || !x.boundary(e)) return 0;
    r = "number " + cardinal(x.s.substr(j, e - j));
    return e;
  });
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i) || !x.starts(i, "No.")) return 0;
    const size_t j = i + 3 + (x.is_space(i + 3) ? 1 : 0), e = x.digits_end(j);
    if (e == j || !x.boundary(e)) return 0;
    r = "number " + cardinal(x.s.substr(j, e - j));
    return e;
  });
  // symbols between whitespace, one pass each: (?<=\s)=(?=\s), + , ×, &
  for (const auto& [sym, name] : {std::pair<const char*, const char*>{"=", "equals"}, {"+", "plus"},
                                  {"\xC3\x97", "times"}, {"&", "and"}}) {
    const std::string_view symbol(sym), word_(name);
    t = pass(t, [&](const Text& x, size_t i, std::string& r) -> size_t {
      if (i == 0 || !space(x.s[i - 1]) || !x.starts(i, symbol) || !x.is_space(i + symbol.size())) return 0;
      r = word_;
      return i + symbol.size();
    });
  }
  // negative numbers: (?<![\w.])[-−](?=\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t s = minus_sign(x, i);
    if (!s || !x.is_digit(i + s)) return 0;
    r = "minus ";
    return i + s;
  });
  // decimals: (?<![\w.])(\d+)\.(\d+)\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t a = x.digits_end(i);
    if (a == i || x.at(a) != '.') return 0;
    const size_t b = x.digits_end(a + 1);
    if (b == a + 1 || !x.boundary(b)) return 0;
    r = number(x.s.substr(i, b - i));
    return b;
  });

  // integers: (?<![\w.])\d{1,3}(?:,\d{3})+(?![\d,]*\d) | (?<![\w.,])\d+(?![\d,]*\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    // (?![\d,]*\d) at e: the run of digits and commas from e holds no digit
    auto clear = [&](size_t e) { while (x.at(e) == ',') ++e; return !x.is_digit(e); };
    size_t end = 0;
    const size_t de = x.digits_end(i);
    if (de == i) return 0;
    if (not_after(x, i, ".", true) && de - i <= 3) {
      size_t e = de, groups = 0;
      while (x.at(e) == ',' && x.digits_end(e + 1) >= e + 4) e += 4, ++groups;
      if (groups && clear(e)) end = e;
    }
    if (!end && not_after(x, i, ".,", true) && clear(de)) end = de;
    if (!end) return 0;
    const std::string_view s = x.s.substr(i, end - i);
    if (s.find(',') != std::string_view::npos) { r = cardinal(without_commas(s)); return end; }
    if ((s.size() > 1 && s[0] == '0') || s.size() >= 7) { r = spell_digits(s); return end; }
    const auto window = x.s.substr(back_code_points(x.s, i, 30), i - back_code_points(x.s, i, 30));
    const auto words = letter_runs(window);
    bool zip = false;
    for (size_t k = words.size() > 3 ? words.size() - 3 : 0; k < words.size(); ++k) {
      const auto w = ascii_lower(words[k]);
      zip = zip || w == "zip" || w == "postal";
    }
    if (s == "911" || (s.size() == 5 && zip)) { r = spell_digits(s); return end; }
    const long n = small(s);
    if (s.size() == 4 && n >= 1000 && n <= 2099) {
      static const char* const cues[] = {"in", "since", "by", "from", "until", "till", "of", "during", "year", "circa",
                                         "before", "after", "around", "early", "late", "mid", "spring", "summer",
                                         "fall", "autumn", "winter", "through", "to"};
      const size_t before_end = python_rstrip(x.s, i);
      const std::string_view before = x.s.substr(0, before_end);
      const size_t b20 = back_code_points(before, before.size(), 20);
      const auto prev = letter_runs(before.substr(b20));
      bool year = false;
      if (!prev.empty()) {
        const auto w = ascii_lower(prev.back());
        for (const char* c : cues) year = year || w == c;
      }
      if (!year) {  // (January|...|December) \w+,?$ anywhere at the end of `before`
        std::string_view b = before;
        for (int comma = 0; comma < 2 && !year; ++comma) {
          if (comma) { if (b.empty() || b.back() != ',') break; b.remove_suffix(1); }
          size_t w = b.size();
          while (w > 0 && word(b[w - 1])) --w;
          if (w == b.size() || w == 0 || b[w - 1] != ' ') continue;
          for (const char* month : kMonths)
            if (std::string_view(b.substr(0, w - 1)).size() >= std::string_view(month).size() &&
                b.substr(0, w - 1).substr(w - 1 - std::string_view(month).size()) == month)
              year = true;
        }
      }
      if (!year) {  // re.match(r"\s*[.,;:!?]|\s*$", t[end:end+2]) and no digit in before[-3:]
        size_t stop = end;  // two code points after the number
        for (int k = 0; k < 2 && stop < x.n(); ++k) {
          ++stop;
          while (stop < x.n() && cont(x.s[stop])) ++stop;
        }
        const std::string_view after = x.s.substr(end, stop - end);
        bool punct = false;
        {
          size_t q = 0;
          while (q < after.size() && space(after[q])) ++q;
          punct = q < after.size() && std::string_view(".,;:!?").find(after[q]) != std::string_view::npos;
          if (!punct) punct = q == after.size();
        }
        bool digit_before = false;
        for (char c : before.substr(back_code_points(before, before.size(), 3))) digit_before = digit_before || digit(c);
        year = punct && !digit_before;
      }
      if (year) { r = pairs(n); return end; }
    }
    r = cardinal(s);
    return end;
  });

  // titles and abbreviations: (?<![\w.])(keys, longest first)\.(?=[\s,;:!?)]|$)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const std::pair<const char*, const char*> titles[] = {
        {"vs", "versus"}, {"Mr", "Mister"}, {"Mrs", "Missus"}, {"Ms", "Miz"}, {"Dr", "Doctor"}, {"Prof", "Professor"},
        {"Mt", "Mount"}, {"Jr", "Junior"}, {"Sr", "Senior"}, {"Gov", "Governor"}, {"Sen", "Senator"},
        {"Rep", "Representative"}, {"Gen", "General"}, {"Capt", "Captain"}, {"Lt", "Lieutenant"}, {"Sgt", "Sergeant"}};
    static const std::pair<const char*, const char*> others[] = {
        {"etc", "et cetera"}, {"e.g", "for example"}, {"i.e", "that is"}, {"approx", "approximately"},
        {"Apt", "Apartment"}, {"Dept", "Department"}, {"Inc", "Incorporated"}, {"Corp", "Corporation"},
        {"Co", "Company"}, {"Ave", "Avenue"}, {"Blvd", "Boulevard"}, {"Rd", "Road"}, {"est", "established"},
        {"min", "minutes"}, {"max", "maximum"}, {"govt", "government"}, {"St", ""}};
    if (!not_after(x, i, ".", true)) return 0;
    // the longest key that matches with its period and look-ahead (keys of equal length cannot both match)
    const char* best = nullptr;
    const char* expansion = nullptr;
    bool title = false;
    auto consider = [&](const char* key, const char* full, bool is_title) {
      const size_t e = i + std::string_view(key).size();
      if (!x.starts(i, key) || x.at(e) != '.') return;
      const size_t f = e + 1;
      const bool ok = f == x.n() || x.is_space(f) || std::string_view(",;:!?)").find(x.at(f)) != std::string_view::npos;
      if (ok && (!best || std::string_view(key).size() > std::string_view(best).size()))
        best = key, expansion = full, title = is_title;
    };
    for (const auto& [k, v] : titles) consider(k, v, true);
    for (const auto& [k, v] : others) consider(k, v, false);
    if (!best) return 0;
    const std::string_view key(best);
    const size_t end = i + key.size() + 1;
    auto name_follows = [&] { const size_t j = x.spaces_end(end); return j > end && upper(x.at(j)); };
    if (key == "St") r = name_follows() ? "Saint" : std::string("Street") + (sentence_end(x, end, true) ? "." : "");
    else if (key == "Dr" && !name_follows()) r = std::string("Drive") + (sentence_end(x, end, true) ? "." : "");
    else if (title) r = expansion;
    else r = std::string(expansion) + (sentence_end(x, end, true) ? "." : "");
    return end;
  });
  // (?<![\w.])vs(?=\s)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true) || !x.starts(i, "vs") || !x.is_space(i + 2)) return 0;
    r = "versus";
    return i + 2;
  });
  // [ \t]{2,} -> " "
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    size_t j = i;
    while (x.at(j) == ' ' || x.at(j) == '\t') ++j;
    if (j - i < 2) return 0;
    r = " ";
    return j;
  });
  return t;
}

}  // namespace lfm::speech_normalize
