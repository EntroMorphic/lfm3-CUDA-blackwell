#pragma once
// Loopback HTTP text serving for liquid-lfm3-server: a native port of tools/lfm3_server.py (Service,
// LocalHTTPServer, Handler) and the request rules it takes from tools/lfm3_chat.py and
// tools/lfm3_worker_client.py. Same endpoints, statuses, error codes, admission and lease rules, and
// response bodies byte for byte (Python json.dumps(ensure_ascii=False, separators=(",", ":"))).
//
// Host-only: the engine is behind Backend (one resident-protocol line in, one reply line out, called
// only from the engine-owner thread), so this layer is tested on any host against a fake engine.
//
// Deliberate differences from the Python server, each at a boundary no conforming client reaches
// (tests/test_lfm3_http_framing.py and tools/lfm3_http_diff.py measure the rest as identical):
//  - Number literals that underflow a double (|x| < 2.2e-308, e.g. 1e-400) are rejected; Python reads
//    them as 0.0 or a subnormal. lfm::Json's existing rule.
//  - Where http.server answers with an HTML error page, or with a body and no status line (HTTP/0.9
//    style, e.g. two-word request lines and HTTP/2.0), the native server sends a status line and a JSON
//    error. Statuses are the same except that HTTP/2.0 gets 505 with a status line. Unsupported methods
//    get 501 as JSON.
//  - A frontend failure that is not a validation error (none is known) answers 400 instead of
//    dropping the connection.
//  - Text is decoded by the engine, outside the frontend lock, so a request arriving during another
//    request's decode is not refused with 429 frontend_busy.
//  - Shutdown requests cancellation of running generations at their next token boundary instead of
//    waiting for them to finish; a second SIGTERM/SIGINT exits at once.
//  - After a fatal engine error, including an engine call longer than engine_timeout, the process
//    keeps answering 503 as the Python server does, but keeps its GPU allocations until it exits (the
//    engine is in-process; there is no worker to kill).
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "lfm/lfm3_text.hpp"
#include "lfm/speech_text.hpp"
#include "lfm/model_config.hpp"

namespace lfm::http {

using Clock = std::chrono::steady_clock;
inline constexpr size_t kMaxBody = 4 * 1024 * 1024;
inline constexpr size_t kMaxResponse = 4 * 1024 * 1024;
// POST /speech (#2): the WAV travels base64 in the JSON, so its replies (engine frame and HTTP body) get a larger
// limit: 64 MiB of base64 holds about 17.5 minutes of 24 kHz 16-bit audio, far more than the step budget allows.
// kSpeechCapacity is the engine's per-chunk capacity (WorkerCore::kSpeechCapacity). The engine bounds a request's
// work itself (WorkerCore::kSpeechStepBudget, 6144 decode steps across all chunks, then truncation); this layer bounds
// the rest of it: at most kSpeechMaxChunks chunks and kSpeechMaxPromptTokens prompt tokens (prefill), and an engine
// request frame within kMaxBody. Worst case, at the slower rates assumed (34 frames/s, 4,700 prefill tokens/s,
// 0.035 s per chunk): 6144/34 + 512 x 0.035 + 131072/4700 = about 227 s, under 80% of the default engine timeout
// (validation/speech-server-v1).
inline constexpr size_t kMaxSpeechResponse = 64 * 1024 * 1024;
inline constexpr int kSpeechCapacity = 4096;
inline constexpr size_t kSpeechMaxChunks = 512;
inline constexpr size_t kSpeechMaxPromptTokens = 131072;
inline constexpr double kSpeechMinEngineTimeout = 300;  // seconds; /speech is refused below it
inline constexpr int kAudioOutToken = 125032;

// An HTTP-visible refusal: status and error code, exactly as lfm3_server.APIError.
struct ApiError : std::exception {
  int status;
  std::string code;
  ApiError(int s, std::string c) : status(s), code(std::move(c)) {}
  const char* what() const noexcept override { return code.c_str(); }
};
// Engine/transport/protocol failure (lfm3_worker_client.WorkerFailure): no cache state is trusted.
struct WorkerFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------------ JSON out (Python json.dumps)

inline void append_py_string(std::string& o, std::string_view s) {
  static const char* hex = "0123456789abcdef";
  o += '"';
  for (const unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; }
        else o += static_cast<char>(c);
    }
  }
  o += '"';
}

// Builds one JSON object in insertion order with compact separators.
class Obj {
 public:
  Obj& str(std::string_view k, std::string_view v) { key(k); append_py_string(s_, v); return *this; }
  Obj& num(std::string_view k, long long v) { key(k); s_ += std::to_string(v); return *this; }
  Obj& boolean(std::string_view k, bool v) { key(k); s_ += v ? "true" : "false"; return *this; }
  Obj& ids(std::string_view k, const std::vector<int>& v) {
    key(k);
    s_ += '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) s_ += ','; s_ += std::to_string(v[i]); }
    s_ += ']';
    return *this;
  }
  Obj& raw(std::string_view k, std::string_view json) { key(k); s_ += json; return *this; }
  std::string done() const { return s_ + "}"; }

 private:
  void key(std::string_view k) { s_ += first_ ? "" : ","; first_ = false; append_py_string(s_, k); s_ += ':'; }
  std::string s_ = "{";
  bool first_ = true;
};

inline std::string error_body(std::string_view code) { return Obj().str("error", code).done(); }

// A double as the engine's JSON reader gets it back exactly (17 significant digits round-trip).
inline std::string exact_number(double v) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

// A finite double as Python's json.dumps writes it (float.__repr__): the shortest digits that round-trip, in fixed
// notation for decimal exponents -4..15 ("0.0001", "1.0", "123.5") and otherwise "d[.ddd]e[+-]XX" ("1e-05",
// "1.5e+16"). Responses carry floats only from /speech (seconds).
inline std::string py_float(double v) {
  if (!std::isfinite(v)) throw std::invalid_argument("non-finite number in a response");
  if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest round-trip
  std::string sci(buf, r.ptr);
  std::string sign;
  if (sci[0] == '-') { sign = "-"; sci.erase(0, 1); }
  const size_t epos = sci.find('e');
  const int exp = std::stoi(sci.substr(epos + 1));
  std::string digits = sci.substr(0, epos);
  if (digits.size() > 1) digits.erase(1, 1);  // drop the '.'
  std::string out;
  if (exp >= -4 && exp < 16) {
    if (exp < 0) {
      out = "0." + std::string(size_t(-exp - 1), '0') + digits;
    } else if (size_t(exp) + 1 >= digits.size()) {
      out = digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
    }
  } else {
    out = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
    char e[16];
    std::snprintf(e, sizeof e, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    out += e;
  }
  return sign + out;
}

// ------------------------------------------------------------------ JSON in (strict_json, integer, ...)

// lfm3_worker_client.strict_json: strict UTF-8, nesting at most 64, no duplicate keys, no NaN/Infinity,
// finite numbers, an object at the top.
inline Json strict_json(std::string_view raw) {
  if (!lfm3_text::valid_utf8(raw)) throw std::invalid_argument("invalid UTF-8");
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

inline const Json::Object& object_of(const Json& j) {
  if (const auto* o = std::get_if<Json::Object>(&j.value)) return *o;
  throw std::invalid_argument("expected object");
}
inline bool is_int(const Json& j) { return std::holds_alternative<double>(j.value) && j.integral; }
// lfm3_chat.integer: a JSON integer (not a float, not a boolean) in low..high.
inline long long integer(const Json& j, long long low, long long high) {
  if (!is_int(j)) throw std::invalid_argument("expected integer");
  const double v = std::get<double>(j.value);
  if (!(v >= double(low) && v <= double(high))) throw std::invalid_argument("integer out of range");
  return static_cast<long long>(v);
}
inline std::set<std::string> keys_of(const Json::Object& o) {
  std::set<std::string> k;
  for (const auto& [key, value] : o) k.insert(key);
  return k;
}

struct Sampling {
  bool do_sample = false;
  double temperature = 1.0, top_p = 1.0, min_p = 0.0, repetition_penalty = 1.0;
  long long top_k = 0, seed = 0;
};

// lfm3_chat.sampling_options.
inline Sampling sampling_options(const Json* options, int vocabulary) {
  Sampling s;
  if (!options) return s;
  static const std::set<std::string> known = {"do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed"};
  const auto& o = object_of(*options);
  for (const auto& [key, value] : o)
    if (!known.count(key)) throw std::invalid_argument("unknown sampling option");
  if (o.count("do_sample")) {
    const auto* b = std::get_if<bool>(&o.at("do_sample").value);
    if (!b) throw std::invalid_argument("do_sample must be boolean");
    s.do_sample = *b;
  }
  if (o.count("top_k")) s.top_k = integer(o.at("top_k"), 0, vocabulary);
  if (o.count("seed")) s.seed = integer(o.at("seed"), 0, 2147483647LL);
  const auto real = [&](const char* key, double& out) {
    if (!o.count(key)) return;
    const auto* d = std::get_if<double>(&o.at(key).value);
    if (!d) throw std::invalid_argument(std::string("invalid ") + key);
    out = *d;
  };
  real("temperature", s.temperature);
  real("top_p", s.top_p);
  real("min_p", s.min_p);
  real("repetition_penalty", s.repetition_penalty);
  for (const auto& [key, value] : {std::pair<const char*, double>{"temperature", s.temperature}, {"top_p", s.top_p},
                                   {"min_p", s.min_p}, {"repetition_penalty", s.repetition_penalty}}) {
    const std::string k = key;
    if (!std::isfinite(value) || value < 0 || (k != "min_p" && value == 0) || ((k == "top_p" || k == "min_p") && value > 1))
      throw std::invalid_argument("invalid " + k);
  }
  return s;
}

inline bool valid_session_id(std::string_view sid) {
  if (sid.size() != 32) return false;
  for (const char c : sid)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
  return true;
}
inline std::string session_id_of(const Json& j) {
  const auto* s = std::get_if<std::string>(&j.value);
  if (!s || !valid_session_id(*s)) throw std::invalid_argument("invalid session ID");
  return *s;
}

// secrets.token_urlsafe(24).
inline std::string token_urlsafe24() {
  unsigned char b[24];
  size_t got = 0;
  while (got < sizeof b) {
    const auto n = ::getrandom(b + got, sizeof b - got, 0);
    if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("getrandom failed"); }
    got += size_t(n);
  }
  static const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  std::string out;
  for (size_t i = 0; i < sizeof b; i += 3) {
    const unsigned v = (unsigned(b[i]) << 16) | (unsigned(b[i + 1]) << 8) | b[i + 2];
    out += a[v >> 18]; out += a[(v >> 12) & 63]; out += a[(v >> 6) & 63]; out += a[v & 63];
  }
  return out;
}

inline bool valid_token(std::string_view t) {
  if (t.size() < 32 || t.size() > 256) return false;
  for (const char c : t)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
  return true;
}

// lfm3_server.token_file: an owned, regular, mode-0600 file of at most 257 bytes.
inline std::string token_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open token file");
  struct stat info {};
  std::string raw(258, '\0');
  ssize_t n = -1;
  if (::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && (info.st_mode & 07777) == 0600 && info.st_uid == ::getuid() &&
      info.st_size <= 257) {
    size_t got = 0;
    for (;;) {
      const auto r = ::read(fd, raw.data() + got, raw.size() - got);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) { n = r < 0 ? -1 : ssize_t(got); break; }
      got += size_t(r);
      if (got == raw.size()) { n = ssize_t(got); break; }
    }
  } else {
    ::close(fd);
    throw std::runtime_error("token file must be owned regular mode-0600 file");
  }
  ::close(fd);
  if (n < 0) throw std::runtime_error("cannot read token file");
  raw.resize(size_t(n));
  for (const unsigned char c : raw)
    if (c >= 0x80) throw std::runtime_error("token file must be ASCII");
  if (!raw.empty() && raw.back() == '\n') raw.pop_back();
  if (!valid_token(raw)) throw std::runtime_error("bearer token must be 32..256 URL-safe characters");
  return raw;
}

// ------------------------------------------------------------------ engine boundary

// The resident protocol (liquid-lfm3-worker, protocol 2) behind an in-process call.
class Backend {
 public:
  virtual ~Backend() = default;
  // One request line (no newline) in, one reply line out. Engine-owner thread only.
  virtual std::string call(const std::string& line) = 0;
  // Write end of the engine's nonblocking cancellation pipe (8-byte little-endian frames).
  virtual int cancellation_fd() const = 0;
  // After a fatal failure: refuse every later call. Engine-owner thread only, or after it stopped.
  virtual void close() = 0;
};

struct Settings {
  int max_sessions = 2, total_capacity = 32768, queue_size = 4, handlers = 8;
  double request_timeout = 330, socket_timeout = 10, session_idle_timeout = 300;
  // One engine call (lfm3_worker_client's per-RPC worker timeout): past it the service fails closed.
  double engine_timeout = 300;
  void validate() const {
    const auto in = [](long long v, long long lo, long long hi) { return v >= lo && v <= hi; };
    if (!in(max_sessions, 1, 2) || !in(total_capacity, 2, 32768) || !in(queue_size, 1, 64) || !in(handlers, 1, 128))
      throw std::invalid_argument("invalid server settings");
    for (const double t : {request_timeout, socket_timeout, session_idle_timeout, engine_timeout})
      if (!std::isfinite(t) || !(t > 0 && t <= 86400)) throw std::invalid_argument("invalid timeout");
  }
};

// lfm3_chat.policy: output vocabulary, modality placeholders and stop ids (sorted, unique).
struct Policy {
  int vocabulary = 0;
  std::vector<int> suppress, stops;
  bool speech = false;  // the engine can run op "speech" (WorkerCore::speech_supported); /speech is refused otherwise
};

inline std::chrono::nanoseconds seconds(double s) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(s));
}

// ------------------------------------------------------------------ Service

class Service {
 public:
  using Event = std::shared_ptr<std::atomic<bool>>;

  Service(Backend& backend, const lfm3_text::Tokenizer& tokenizer, Policy policy, Settings settings)
      : backend_(backend), tokenizer_(tokenizer), policy_(std::move(policy)), settings_(settings) {
    settings_.validate();
    owner_ = std::thread([this] {
      run();
      std::lock_guard g(done_lock_);
      owner_done_ = true;
      done_cv_.notify_all();
    });
    watchdog_ = std::thread([this] { watch(); });
  }
  ~Service() {
    stopping_ = true;
    {
      std::lock_guard g(done_lock_);
      watchdog_stop_ = true;
      done_cv_.notify_all();
    }
    if (watchdog_.joinable()) watchdog_.join();
    if (owner_.joinable()) owner_.join();
  }
  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;

  const Settings& settings() const { return settings_; }
  bool ready() {
    std::lock_guard g(lock_);
    return !failed_ && !stopping_;
  }
  bool failed() {
    std::lock_guard g(lock_);
    return failed_;
  }
  bool cleanup_failed() {
    std::lock_guard g(lock_);
    return cleanup_failed_;
  }
  size_t session_count() {
    std::lock_guard g(lock_);
    return sessions_.size();
  }

  std::string create(const Json& body) {
    const auto& o = object_of(body);
    if (keys_of(o) != std::set<std::string>{"capacity"}) throw std::invalid_argument("create requires capacity only");
    const int capacity = int(integer(o.at("capacity"), 2, settings_.total_capacity));
    const std::string sid = token_urlsafe24();
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      int reserved = 0;
      for (const auto& [id, s] : sessions_) reserved += s.capacity;
      if (int(sessions_.size()) >= settings_.max_sessions || reserved + capacity > settings_.total_capacity)
        throw ApiError(429, "session_quota");
      sessions_[sid] = Session{capacity, true, "create", nullptr, Clock::now()};
    }
    auto created = std::make_shared<bool>(false);
    auto run = [this, sid, capacity, created] {
      const auto result = rpc(Obj().str("op", "create").str("session_id", sid).num("capacity", capacity));
      const auto* id = result.has("session_id") ? std::get_if<std::string>(&result.at("session_id").value) : nullptr;
      if (!id || *id != sid) throw WorkerFailure("session mismatch");
      *created = true;
      return Obj().str("session_id", sid).num("capacity", capacity).boolean("gpu_batching", false).done();
    };
    auto cleanup = [this, sid, created] {
      std::lock_guard g(lock_);
      auto it = sessions_.find(sid);
      if (it == sessions_.end()) return;
      if (*created) { it->second.busy = false; it->second.operation.clear(); it->second.last_used = Clock::now(); }
      else sessions_.erase(it);
    };
    return submit(run, cleanup, [this, sid] { close_allocation(sid); });
  }

  std::string cancel(const std::string& sid, const Json& body) {
    if (!object_of(body).empty()) throw std::invalid_argument("cancel requires an empty object");
    if (!valid_session_id(sid)) throw std::invalid_argument("invalid session ID");
    std::lock_guard g(lock_);
    if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
    auto it = sessions_.find(sid);
    if (it == sessions_.end()) throw ApiError(404, "unknown_session");
    const auto& s = it->second;
    if (!s.busy || s.operation != "generate" || !s.cancel_event) throw ApiError(409, "generation_not_active");
    s.cancel_event->store(true);
    // Nonblocking targeted pipe notification only. A failed notification is not acknowledged; the event
    // stays set, and only the owner completes the generation or reset.
    try { send_cancel(sid); }
    catch (const WorkerFailure&) { throw ApiError(503, "cancellation_transport_failed"); }
    return Obj().str("session_id", sid).boolean("cancel_requested", true).boolean("gpu_batching", false).done();
  }

  std::string lifecycle(const std::string& sid, const std::string& op, const Json& body) {
    if ((op != "reset" && op != "close") || !object_of(body).empty()) throw std::invalid_argument("invalid lifecycle request");
    lease(sid, op, nullptr);
    auto run = [this, sid, op] {
      const auto result = rpc(Obj().str("op", op).str("session_id", sid));
      if (!flag(result, op == "reset" ? "reset" : "closed") || !session_matches(result, sid))
        throw WorkerFailure("invalid lifecycle reply");
      if (op == "close") {
        std::lock_guard g(lock_);
        sessions_.erase(sid);
      }
      return Obj().boolean("ok", true).done();
    };
    return submit(run, [this, sid] { release(sid, nullptr); }, nullptr);
  }

  // `body` is shared, not copied, and dropped as soon as the prompt is encoded.
  std::string generate(std::shared_ptr<const Json> body) {
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    auto request = std::make_shared<Generation>(validate_generation(*body, body));
    body.reset();
    const auto sid = request->sid;
    Event event = std::make_shared<std::atomic<bool>>(false);
    const int capacity = lease(sid, "generate", event);
    try {
      frontend_call([&] { request->input = encode(*request, capacity); }, deadline);
      request->drop_body();
    } catch (...) {
      release(sid, event);
      throw;
    }
    auto run = [this, request, event, deadline] {
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      if (event->load()) {
        const auto reset = rpc(Obj().str("op", "reset").str("session_id", request->sid));
        if (!flag(reset, "reset") || !session_matches(reset, request->sid)) throw WorkerFailure("invalid cancellation reset reply");
        return Obj().str("session_id", request->sid).str("text", "").ids("generated_ids", {}).num("consumed_tokens", 0)
            .num("reused_tokens", 0).boolean("cancelled", true).str("finish_reason", "cancelled").boolean("cache_valid", false)
            .boolean("do_sample", request->sampling.do_sample).num("seed", request->sampling.seed).boolean("gpu_batching", false)
            .done();
      }
      const auto result = rpc(native_request(*request).str("op", "generate").str("session_id", request->sid),
                              {{request->sid, event}});
      auto response = generation_response(*request, result, *event, false);
      const auto out = response.body.done();
      if (out.size() > kMaxResponse) throw WorkerFailure("response exceeds limit");
      return out;
    };
    return submit(run, [this, sid, event] { release(sid, event); }, nullptr, deadline);
  }

  std::string generate_batch(std::shared_ptr<const Json> body) {
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    const auto& o = object_of(*body);
    const auto* list = o.size() == 1 && o.count("requests") ? std::get_if<Json::Array>(&o.at("requests").value) : nullptr;
    if (!list || list->size() != 2) throw std::invalid_argument("batch requires exactly two requests");
    auto requests = std::make_shared<std::array<Generation, 2>>();
    for (size_t i = 0; i < 2; ++i) (*requests)[i] = validate_generation((*list)[i], body);
    body.reset();
    const std::array<std::string, 2> sids{(*requests)[0].sid, (*requests)[1].sid};
    if (sids[0] == sids[1]) throw std::invalid_argument("duplicate batch session");
    const std::array<Event, 2> events{std::make_shared<std::atomic<bool>>(false), std::make_shared<std::atomic<bool>>(false)};
    std::array<int, 2> capacities{};
    // Check both before mutating either lease: never release somebody else's busy lease when the second
    // lane cannot be admitted.
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      for (const auto& sid : sids)
        if (!sessions_.count(sid)) throw ApiError(404, "unknown_session");
      for (const auto& sid : sids)
        if (sessions_.at(sid).busy) throw ApiError(409, "session_busy");
      for (size_t i = 0; i < 2; ++i) {
        auto& s = sessions_.at(sids[i]);
        capacities[i] = s.capacity;
        s.busy = true; s.operation = "generate"; s.cancel_event = events[i];
      }
    }
    auto cleanup = [this, sids, events] { for (size_t i = 0; i < 2; ++i) release(sids[i], events[i]); };
    try {
      frontend_call([&] {
        for (size_t i = 0; i < 2; ++i) {
          if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
          (*requests)[i].input = encode((*requests)[i], capacities[i]);
        }
      }, deadline);
      for (auto& r : *requests) r.drop_body();
    } catch (...) {
      cleanup();
      throw;
    }
    auto run = [this, requests, events, deadline] {
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      std::string lanes = "[";
      for (size_t i = 0; i < 2; ++i) {
        if (i) lanes += ',';
        lanes += native_request((*requests)[i]).str("session_id", (*requests)[i].sid).done();
      }
      lanes += ']';
      const auto result = rpc(Obj().str("op", "generate_batch").raw("lanes", lanes),
                              {{(*requests)[0].sid, events[0]}, {(*requests)[1].sid, events[1]}}, true);
      const auto* gpu = result.has("gpu_batching") ? std::get_if<bool>(&result.at("gpu_batching").value) : nullptr;
      const long long paired = result.has("paired_decode_calls") && is_int(result.at("paired_decode_calls"))
                                   ? static_cast<long long>(std::get<double>(result.at("paired_decode_calls").value)) : -1;
      if (paired < 0 || !gpu || *gpu != (paired > 0)) throw WorkerFailure("invalid batch accounting");
      const auto* results = result.has("results") ? std::get_if<Json::Array>(&result.at("results").value) : nullptr;
      if (!results || results->size() != 2) throw WorkerFailure("invalid batch results");
      std::vector<Response> responses;
      for (size_t i = 0; i < 2; ++i) {
        const auto& lane = (*results)[i];
        const auto* lo = std::get_if<Json::Object>(&lane.value);
        const auto boolean_field = [&](const char* k) { return lo && lo->count(k) && std::holds_alternative<bool>(lo->at(k).value); };
        if (!lo || !boolean_field("gpu_batching") || (lo->count("ok") && !flag(lane, "ok"))
            || std::get<bool>(lo->at("gpu_batching").value) != *gpu || !lo->count("paired_decode_calls")
            || !is_int(lo->at("paired_decode_calls")) || std::get<double>(lo->at("paired_decode_calls").value) != double(paired)
            || !boolean_field("cancelled") || !boolean_field("cache_valid"))
          throw WorkerFailure("invalid lane accounting");
        auto response = generation_response((*requests)[i], lane, *events[i], *gpu);
        response.body.num("paired_decode_calls", paired);
        responses.push_back(std::move(response));
      }
      // Prefill samples the first token without a decode forward, so each paired forward needs another
      // output slot in both lanes. Cancellation may discard the token sampled after that forward.
      long long budget = std::numeric_limits<long long>::max();
      for (const auto& r : *requests) budget = std::min<long long>(budget, r.count - 1);
      if (paired > budget) throw WorkerFailure("paired calls exceed requested decode budget");
      bool any_cancelled = false;
      long long completed = std::numeric_limits<long long>::max();
      for (const auto& r : responses) {
        const long long maximum = (long long)r.generated.size() - (r.cancelled ? 0 : 1);
        if (paired > maximum) throw WorkerFailure("paired calls exceed lane decode work");
        any_cancelled = any_cancelled || r.cancelled;
        completed = std::min<long long>(completed, (long long)r.generated.size() - 1);
      }
      if (!any_cancelled && paired != completed) throw WorkerFailure("paired calls do not match completed group");
      std::string items = "[";
      for (size_t i = 0; i < 2; ++i) {
        const auto lane = responses[i].body.done();
        if (lane.size() > kMaxResponse) throw WorkerFailure("response exceeds limit");
        if (i) items += ',';
        items += lane;
      }
      items += ']';
      const auto out = Obj().raw("results", items).boolean("gpu_batching", *gpu).num("paired_decode_calls", paired).done();
      if (out.size() > kMaxResponse) throw WorkerFailure("response exceeds limit");
      return out;
    };
    return submit(run, cleanup, nullptr, deadline);
  }

  // Stops admission, fails every later request, cancels running generations at their next token
  // boundary and waits for the engine owner to finish its queue.
  void close() {
    {
      std::lock_guard g(lock_);
      stopping_ = true;
      failed_ = true;
      // A requested cancellation, exactly as /cancel requests it, so the reply is a solicited one.
      for (auto& [sid, s] : sessions_)
        if (s.busy && s.operation == "generate" && s.cancel_event) s.cancel_event->store(true);
      sessions_.clear();
    }
    {
      std::lock_guard g(active_lock_);
      for (auto& [sid, active] : active_) {
        if (active.sent) continue;
        try { write_frame(active.token); active.sent = true; }
        catch (const WorkerFailure&) {}  // the generation then finishes normally; shutdown still waits for it
      }
    }
    queue_cv_.notify_all();
    // As lfm3_server.Service.close: a bounded wait for the engine owner. An engine call that never
    // returns cannot be reclaimed in-process; the caller exits instead (cleanup_failed()).
    bool done;
    {
      std::unique_lock g(done_lock_);
      done = done_cv_.wait_for(g, seconds(settings_.engine_timeout + 5), [&] { return owner_done_; });
      watchdog_stop_ = true;
      done_cv_.notify_all();
    }
    if (watchdog_.joinable()) watchdog_.join();
    if (!done) {
      owner_.detach();
      std::lock_guard g(lock_);
      cleanup_failed_ = true;
      return;
    }
    owner_.join();
    fatal();
  }

 private:
  struct Session {
    int capacity = 0;
    bool busy = false;
    std::string operation;
    Event cancel_event;
    Clock::time_point last_used;
  };
  struct Generation {
    std::string sid;
    const Json* messages = nullptr;
    std::shared_ptr<const Json> body;  // the parsed request; keeps `messages` alive until encoding
    int count = 0;
    Sampling sampling;
    std::vector<int> input;
    void drop_body() { messages = nullptr; body.reset(); }
  };
  struct Speech {
    std::shared_ptr<const Json> body;  // keeps `messages` alive until encoding
    const Json* messages = nullptr;
    std::string mode;
    int count = 0;
    long long seed = 0;
    bool split = false;
    std::vector<std::vector<int>> prompts;
    std::vector<std::string> texts;  // each chunk's text, when split
    std::string chunks_json;         // the prompts as the engine request carries them
  };
  Speech validate_speech(const Json& item) const {
    const auto& o = object_of(item);
    static const std::set<std::string> allowed = {"messages", "mode", "max_new_tokens", "seed", "split"};
    for (const auto& [key, value] : o)
      if (!allowed.count(key)) throw std::invalid_argument("invalid speech fields");
    for (const char* key : {"messages", "mode", "max_new_tokens"})
      if (!o.count(key)) throw std::invalid_argument("invalid speech fields");
    Speech s;
    const auto* mode = std::get_if<std::string>(&o.at("mode").value);
    if (!mode || (*mode != "audio" && *mode != "interleaved")) throw std::invalid_argument("mode must be audio or interleaved");
    s.mode = *mode;
    s.count = int(integer(o.at("max_new_tokens"), 1, kSpeechCapacity - 1));
    s.seed = o.count("seed") ? integer(o.at("seed"), 0, 2147483647) : 0;
    if (o.count("split")) {
      const auto* b = std::get_if<bool>(&o.at("split").value);
      if (!b) throw std::invalid_argument("split must be a boolean");
      s.split = *b;
    } else {
      s.split = s.mode == "audio";
    }
    if (s.split && s.mode != "audio") throw std::invalid_argument("split is for audio mode");
    if (!std::holds_alternative<Json::Array>(o.at("messages").value)) throw std::invalid_argument("messages must be a list");
    s.messages = &o.at("messages");
    (void)lfm3_text::parse_messages(*s.messages);
    return s;
  }
  // Render and tokenize each chunk as a single request (lfm3_text::encode_messages), split by sentences when asked.
  void encode_speech(Speech& s) const {
    const auto messages = lfm3_text::parse_messages(*s.messages);
    const auto chats = s.split ? speech_text::sentence_requests(messages)
                               : std::vector<std::vector<lfm3_text::Message>>{messages};
    if (chats.size() > kSpeechMaxChunks) throw std::invalid_argument("speech request has too many sentences");
    size_t tokens = 0;
    for (const auto& chat : chats) {
      auto ids = lfm3_text::encode_messages(tokenizer_, chat, kSpeechCapacity, s.count);
      for (const int id : ids) {
        if (id < 0 || id >= policy_.vocabulary) throw std::invalid_argument("input_ids out of range");
        if (std::binary_search(policy_.suppress.begin(), policy_.suppress.end(), id))
          throw std::invalid_argument("multimodal placeholder input is unsupported");
      }
      tokens += ids.size();
      if (tokens > kSpeechMaxPromptTokens) throw std::invalid_argument("speech request has too many prompt tokens");
      s.prompts.push_back(std::move(ids));
      if (s.split) s.texts.push_back(chat.back().content);
    }
    // The engine request, built here so that one too large for an engine frame is a 400, not a fatal engine error.
    s.chunks_json = "[";
    for (size_t k = 0; k < s.prompts.size(); ++k) {
      s.chunks_json += k ? ",[" : "[";
      for (size_t i = 0; i < s.prompts[k].size(); ++i) s.chunks_json += (i ? "," : "") + std::to_string(s.prompts[k][i]);
      s.chunks_json += "]";
    }
    s.chunks_json += "]";
    if (s.chunks_json.size() + 256 > kMaxBody) throw std::invalid_argument("speech request exceeds the engine frame");
    s.messages = nullptr;
    s.body.reset();
  }
  std::string speech_response(const Speech& s, const Json& r) const {
    if (!r.has("cancelled") || !std::holds_alternative<bool>(r.at("cancelled").value)) throw WorkerFailure("invalid speech reply");
    if (std::get<bool>(r.at("cancelled").value))
      return Obj().str("mode", s.mode).boolean("cancelled", true).num("seed", s.seed).boolean("split", s.split).done();
    std::vector<int> slots;
    for (const auto& v : r.at("slots").array()) {
      const int id = v.integer();
      if (id != kAudioOutToken) slots.push_back(id);
    }
    const auto& chunks = r.at("chunks").array();
    const bool truncated = r.at("truncated").boolean();
    if (chunks.empty() || chunks.size() > s.prompts.size() || (chunks.size() < s.prompts.size() && !truncated))
      throw WorkerFailure("invalid speech reply");  // fewer chunks only when truncated
    std::string list = "[";
    for (size_t k = 0; k < chunks.size(); ++k) {
      Obj c;
      if (s.split) c.str("text", s.texts[k]);
      c.num("frames", chunks[k].at("frames").integer()).boolean("ended_by_eos", chunks[k].at("ended_by_eos").boolean());
      list += (k ? "," : "") + c.done();
    }
    list += "]";
    const auto out = Obj().str("mode", s.mode).str("text", s.mode == "audio" ? std::string() : tokenizer_.decode(slots, true))
        .str("audio_wav_base64", r.at("wav_base64").string()).num("sample_rate", 24000)
        .raw("audio_seconds", py_float(std::get<double>(r.at("audio_seconds").value)))
        .num("frames", r.at("frames").integer()).raw("chunks", list).num("seed", s.seed).boolean("split", s.split)
        .boolean("cancelled", false).boolean("truncated", truncated).raw("prefill_seconds", py_float(std::get<double>(r.at("prefill_seconds").value)))
        .raw("generate_seconds", py_float(std::get<double>(r.at("generate_seconds").value))).done();
    if (out.size() > kMaxSpeechResponse) throw WorkerFailure("response exceeds limit");
    return out;
  }
  struct Response {
    Obj body;
    std::vector<int> generated;
    bool cancelled = false;
  };
  struct Job {
    std::function<std::string()> function;
    std::promise<std::string> promise;
    Clock::time_point deadline;
    std::function<void()> cleanup, discard;
    std::mutex completion;
    bool abandoned = false;
  };
  struct Active {
    std::uint64_t token = 0;
    bool sent = false;
  };

  static bool flag(const Json& j, const char* key) {
    if (!j.has(key)) return false;
    const auto* b = std::get_if<bool>(&j.at(key).value);
    return b && *b;
  }
  static bool session_matches(const Json& j, const std::string& sid) {
    if (!j.has("session_id")) return true;
    const auto* s = std::get_if<std::string>(&j.at("session_id").value);
    return s && *s == sid;
  }

 public:
  // POST /speech (#2): text in, speech (and, interleaved, text) out. No session: one request at a time on the
  // engine's speech runner, queued and timed like every engine job. Body: {"messages": [...] (the chat, as for
  // /generate), "mode": "audio" | "interleaved", "max_new_tokens": 1..4095 (per chunk), "seed": 0..2^31-1 (default
  // 0), "split": bool (audio only; default true: long-form TTS by sentences, lfm/speech_text.hpp)}. Reply:
  // {"mode", "text" (the emitted text; "" in audio mode), "audio_wav_base64" (24 kHz 16-bit mono WAV),
  // "sample_rate", "audio_seconds", "frames", "chunks": [{"text" (split only), "frames", "ended_by_eos"}], "seed",
  // "split", "cancelled", "truncated" (the step budget stopped it short of its text: chunks skipped, or the last one
  // cut), "prefill_seconds", "generate_seconds"}. The WAV is byte for byte lfm3-speech's for the same request.
  std::string speech(std::shared_ptr<const Json> body) {
    // the step budget's worst case (~227 s) is sized for the default engine timeout; a shorter one could let an
    // admitted request fail the service closed
    if (!policy_.speech || settings_.engine_timeout < kSpeechMinEngineTimeout) throw ApiError(501, "speech_unsupported");
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    auto request = std::make_shared<Speech>(validate_speech(*body));
    request->body = std::move(body);  // `messages` points into it until encode_speech drops it
    frontend_call([&] { encode_speech(*request); }, deadline);
    Event event = std::make_shared<std::atomic<bool>>(false);
    auto run = [this, request, event, deadline] {
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      const auto result = rpc(Obj().str("op", "speech").raw("chunks", request->chunks_json).str("mode", request->mode)
                                  .num("max_new_tokens", request->count).num("seed", request->seed)
                                  .boolean("chunked", request->split),
                              {{"speech", event}}, false, kMaxSpeechResponse);
      return speech_response(*request, result);
    };
    return submit(run, [] {}, nullptr, deadline);
  }

 private:
  // lfm3_server.Service._validate_generation (before any lease): fields, types, ranges, the messages
  // themselves and the sampling options.
  Generation validate_generation(const Json& item, std::shared_ptr<const Json> owner) const {
    const auto& o = object_of(item);
    static const std::set<std::string> allowed = {"session_id", "messages", "max_new_tokens", "sampling"};
    for (const auto& [key, value] : o)
      if (!allowed.count(key)) throw std::invalid_argument("invalid generation fields");
    for (const char* key : {"session_id", "messages", "max_new_tokens"})
      if (!o.count(key)) throw std::invalid_argument("invalid generation fields");
    if (o.count("sampling") && !std::holds_alternative<Json::Object>(o.at("sampling").value))
      throw std::invalid_argument("sampling must be an object");
    if (!std::holds_alternative<Json::Array>(o.at("messages").value)) throw std::invalid_argument("messages must be a list");
    Generation g;
    g.sid = session_id_of(o.at("session_id"));
    g.count = int(integer(o.at("max_new_tokens"), 1, 32768));
    g.body = std::move(owner);
    g.messages = &o.at("messages");
    (void)lfm3_text::parse_messages(*g.messages);
    g.sampling = sampling_options(o.count("sampling") ? &o.at("sampling") : nullptr, policy_.vocabulary);
    return g;
  }

  // lfm3_chat.make_request with the lease's capacity: render, tokenize, fit, placeholder rejection.
  std::vector<int> encode(const Generation& g, int capacity) const {
    const auto messages = lfm3_text::parse_messages(*g.messages);
    auto ids = lfm3_text::encode_messages(tokenizer_, messages, capacity, g.count);
    if (ids.empty() || ids.size() > size_t(capacity) || ids.size() + size_t(g.count) > size_t(capacity))
      throw std::invalid_argument("prompt plus generation exceeds native cache capacity; no truncation performed");
    for (const int id : ids) {
      if (id < 0 || id >= policy_.vocabulary) throw std::invalid_argument("input_ids out of range");
      if (std::binary_search(policy_.suppress.begin(), policy_.suppress.end(), id))
        throw std::invalid_argument("multimodal placeholder input is unsupported in text-only mode");
    }
    return ids;
  }

  // The engine request make_request builds, minus capacity/suppress/stop/long-context; decoded text on.
  static Obj native_request(const Generation& g) {
    const auto& s = g.sampling;
    return Obj().ids("input_ids", g.input).num("max_new_tokens", g.count).boolean("do_sample", s.do_sample)
        .raw("temperature", exact_number(s.temperature)).num("top_k", s.top_k).raw("top_p", exact_number(s.top_p))
        .raw("min_p", exact_number(s.min_p)).raw("repetition_penalty", exact_number(s.repetition_penalty)).num("seed", s.seed)
        .boolean("return_text", true);
  }

  // lfm3_server.Service._generation_response plus the decoded text.
  Response generation_response(const Generation& g, const Json& result, const std::atomic<bool>& event, bool gpu_batching) const {
    const auto* o = std::get_if<Json::Object>(&result.value);
    if (!o || o->count("sampling_policy") || o->count("sampled_policy") || o->count("sampling"))
      throw WorkerFailure("ambiguous generation metadata");
    bool cancelled = false;
    if (o->count("cancelled")) {
      const auto* b = std::get_if<bool>(&o->at("cancelled").value);
      if (!b) throw WorkerFailure("invalid cancellation metadata");
      cancelled = *b;
    }
    const auto* cache_valid = o->count("cache_valid") ? std::get_if<bool>(&o->at("cache_valid").value) : nullptr;
    if (cancelled) {
      if (!event.load() || !cache_valid || *cache_valid) throw WorkerFailure("unsolicited or invalid cancellation");
    } else {
      const auto* finish = o->count("finish_reason") ? std::get_if<std::string>(&o->at("finish_reason").value) : nullptr;
      if ((o->count("cache_valid") && !(cache_valid && *cache_valid)) || (finish && *finish == "cancelled"))
        throw WorkerFailure("unsolicited cache invalidation");
    }
    std::vector<int> generated;
    const auto* ids = o->count("generated_ids") ? std::get_if<Json::Array>(&o->at("generated_ids").value) : nullptr;
    if (!ids || ids->size() > size_t(g.count)) throw WorkerFailure("invalid generation reply");
    for (const auto& id : *ids) {
      if (!is_int(id)) throw WorkerFailure("invalid generation reply");
      const double v = std::get<double>(id.value);
      if (!(v >= 0 && v <= double(policy_.vocabulary - 1))) throw WorkerFailure("invalid generation reply");
      generated.push_back(int(v));
    }
    const auto is_stop = [&](int t) { return std::binary_search(policy_.stops.begin(), policy_.stops.end(), t); };
    bool invalid = !session_matches_required(result, g.sid) || (generated.empty() && !cancelled);
    for (const int t : generated)
      if (std::binary_search(policy_.suppress.begin(), policy_.suppress.end(), t)) invalid = true;
    for (size_t i = 0; i + 1 < generated.size(); ++i)
      if (is_stop(generated[i])) invalid = true;
    if (!cancelled && !generated.empty() && generated.size() < size_t(g.count) && !is_stop(generated.back())) invalid = true;
    if (invalid) throw WorkerFailure("invalid generation reply");
    const auto* do_sample = o->count("do_sample") ? std::get_if<bool>(&o->at("do_sample").value) : nullptr;
    if (!do_sample || *do_sample != g.sampling.do_sample || !o->count("seed") || !is_int(o->at("seed"))
        || std::get<double>(o->at("seed").value) != double(g.sampling.seed))
      throw WorkerFailure("sampling metadata mismatch");
    const long long consumed = cancelled ? 0 : (long long)g.input.size() + (long long)generated.size() - 1;
    const auto exact_int = [&](const char* key, long long lo, long long hi) -> long long {
      if (!o->count(key) || !is_int(o->at(key))) throw WorkerFailure(std::string("invalid ") + key);
      const double v = std::get<double>(o->at(key).value);
      if (!(v >= double(lo) && v <= double(hi))) throw WorkerFailure(std::string("invalid ") + key);
      return (long long)v;
    };
    exact_int("consumed_tokens", consumed, consumed);
    const long long reused = exact_int("reused_tokens", 0, cancelled ? 0 : (long long)g.input.size());
    const auto* text = o->count("text") ? std::get_if<std::string>(&o->at("text").value) : nullptr;
    if (!text) throw WorkerFailure("missing decoded text");
    Response r;
    r.generated = generated;
    r.cancelled = cancelled;
    r.body.str("session_id", g.sid).str("text", *text).ids("generated_ids", generated).num("reused_tokens", reused)
        .num("consumed_tokens", consumed).boolean("do_sample", g.sampling.do_sample).num("seed", g.sampling.seed)
        .boolean("gpu_batching", gpu_batching).boolean("cancelled", cancelled).boolean("cache_valid", !cancelled)
        .str("finish_reason", cancelled ? "cancelled" : is_stop(generated.back()) ? "stop" : "length");
    return r;
  }
  static bool session_matches_required(const Json& j, const std::string& sid) {
    if (!j.has("session_id")) return false;
    const auto* s = std::get_if<std::string>(&j.at("session_id").value);
    return s && *s == sid;
  }

  // One serialized CPU frontend job at a time (lfm3_server.Service._frontend_call, wait=False):
  // a second concurrent encode is refused rather than queued.
  template <class F>
  void frontend_call(F&& function, Clock::time_point deadline) {
    if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
    std::unique_lock guard(frontend_lock_, std::try_to_lock);
    if (!guard.owns_lock()) throw ApiError(429, "frontend_busy");
    if (stopping_) throw ApiError(503, "worker_unavailable");
    if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
    function();
    if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
  }

  int lease(const std::string& sid, const std::string& operation, Event event) {
    if (!valid_session_id(sid)) throw std::invalid_argument("invalid session ID");
    std::lock_guard g(lock_);
    if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
    auto it = sessions_.find(sid);
    if (it == sessions_.end()) throw ApiError(404, "unknown_session");
    if (it->second.busy) throw ApiError(409, "session_busy");
    it->second.busy = true;
    it->second.operation = operation;
    it->second.cancel_event = std::move(event);
    return it->second.capacity;
  }

  void release(const std::string& sid, const Event& event) {
    std::lock_guard g(lock_);
    auto it = sessions_.find(sid);
    if (it != sessions_.end() && (!event || it->second.cancel_event == event)) {
      it->second.busy = false;
      it->second.operation.clear();
      it->second.cancel_event = nullptr;
      it->second.last_used = Clock::now();
    }
  }

  void write_frame(std::uint64_t token) {
    unsigned char frame[8];
    for (int i = 0; i < 8; ++i) frame[i] = static_cast<unsigned char>(token >> (8 * i));
    ssize_t n;
    do n = ::write(backend_.cancellation_fd(), frame, sizeof frame);
    while (n < 0 && errno == EINTR);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) throw WorkerFailure("cancellation control queue full");
    if (n < 0) throw WorkerFailure("cancellation control pipe failed");
    if (n != 8) throw WorkerFailure("partial cancellation control write");
  }

  // lfm3_worker_client.WorkerClient.cancel: request only; the RPC reply acknowledges termination.
  bool send_cancel(const std::string& sid) {
    std::lock_guard g(active_lock_);
    auto it = active_.find(sid);
    if (backend_closed_ || it == active_.end()) return false;
    if (it->second.sent) return true;
    write_frame(it->second.token);
    it->second.sent = true;
    return true;
  }

  // lfm3_worker_client.WorkerClient.call + Service._rpc: owner thread only. Any failure is fatal.
  Json rpc(Obj request, std::vector<std::pair<std::string, Event>> targets = {}, bool batch = false,
           size_t max_reply = kMaxBody) {
    if (backend_closed_) throw WorkerFailure("worker closed");
    if (sequence_ >= 2147483647) throw WorkerFailure("RPC sequence exhausted; restart required");
    const std::uint64_t id = ++sequence_;
    const auto line = request.num("rpc_id", (long long)id).done();
    if (line.size() + 1 > kMaxBody) throw WorkerFailure("request frame exceeds limit");
    {
      std::lock_guard g(active_lock_);
      active_.clear();
      for (size_t i = 0; i < targets.size(); ++i)
        active_[targets[i].first] = Active{batch ? ((std::uint64_t(i + 1) << 32) | id) : id, false};
    }
    struct Clear {
      Service* s;
      ~Clear() { std::lock_guard g(s->active_lock_); s->active_.clear(); }
    } clear{this};
    for (const auto& [sid, event] : targets)
      if (event && event->load() && !send_cancel(sid)) throw WorkerFailure("pre-dispatch cancellation was not admitted");
    std::string reply;
    call_started_ = Clock::now().time_since_epoch().count();
    in_call_ = true;
    try {
      reply = backend_.call(line);
    } catch (const std::exception& e) {
      in_call_ = false;
      throw WorkerFailure(std::string("worker RPC failed: ") + e.what());
    }
    in_call_ = false;
    if (timed_out_) throw WorkerFailure("engine call exceeded the engine timeout");
    while (!reply.empty() && reply.back() == '\n') reply.pop_back();
    if (reply.size() > max_reply) throw WorkerFailure("worker frame exceeds limit");
    Json result;
    try { result = strict_json(reply); }
    catch (const std::exception&) { throw WorkerFailure("invalid worker reply"); }
    const auto* ok = result.has("ok") ? std::get_if<bool>(&result.at("ok").value) : nullptr;
    if (!ok || !result.has("rpc_id") || !is_int(result.at("rpc_id")) || std::get<double>(result.at("rpc_id").value) != double(id))
      throw WorkerFailure("invalid worker reply");
    // Even native execution errors are fail-closed, rather than guessing which CUDA errors might leave
    // the process or another cache usable.
    if (!*ok) throw WorkerFailure("native operation rejected");
    std::get<Json::Object>(result.value).erase("rpc_id");
    return result;
  }

  void close_allocation(const std::string& sid) {
    const auto result = rpc(Obj().str("op", "close").str("session_id", sid));
    if (!flag(result, "closed") || !session_matches(result, sid)) throw WorkerFailure("invalid close reply");
    std::lock_guard g(lock_);
    sessions_.erase(sid);
  }

  void fatal() {
    {
      std::lock_guard g(lock_);
      failed_ = true;
      sessions_.clear();
    }
    bool failed_cleanup = false;
    {
      std::lock_guard g(active_lock_);
      if (backend_closed_) return;
      backend_closed_ = true;
      try { backend_.close(); }
      catch (...) { failed_cleanup = true; }
    }
    if (failed_cleanup) {
      std::lock_guard g(lock_);
      cleanup_failed_ = true;
    }
  }

  std::string submit(std::function<std::string()> function, std::function<void()> cleanup, std::function<void()> discard,
                     std::optional<Clock::time_point> when = std::nullopt) {
    const auto deadline = when ? *when : Clock::now() + seconds(settings_.request_timeout);
    auto job = std::make_shared<Job>();
    job->function = std::move(function);
    job->deadline = deadline;
    job->cleanup = cleanup;
    job->discard = std::move(discard);
    auto future = job->promise.get_future();
    try {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      std::lock_guard q(queue_lock_);
      if (int(queue_.size()) >= settings_.queue_size) throw ApiError(429, "queue_full");
      queue_.push_back(job);
    } catch (...) {
      cleanup();
      throw;
    }
    queue_cv_.notify_all();
    if (future.wait_until(deadline) == std::future_status::ready) return future.get();
    {
      std::lock_guard g(job->completion);
      if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) return future.get();
      job->abandoned = true;
    }
    // Do not release a running generation lease or pretend cancellation.
    throw ApiError(504, "request_deadline_outcome_unknown");
  }

  // lfm3_worker_client's per-RPC deadline: an engine call running past engine_timeout fails the
  // service closed (503 for everything, sessions dropped) and requests cancellation of whatever is
  // running. The late reply is then refused as a failure.
  void watch() {
    std::unique_lock g(done_lock_);
    while (!watchdog_stop_) {
      done_cv_.wait_for(g, std::chrono::milliseconds(100));
      if (watchdog_stop_ || !in_call_ || timed_out_) continue;
      const auto started = Clock::time_point(Clock::duration(call_started_.load()));
      if (Clock::now() - started < seconds(settings_.engine_timeout)) continue;
      timed_out_ = true;
      {
        std::lock_guard l(lock_);
        failed_ = true;
        sessions_.clear();
      }
      std::lock_guard a(active_lock_);
      for (auto& [sid, active] : active_) {
        if (active.sent) continue;
        try { write_frame(active.token); active.sent = true; } catch (const WorkerFailure&) {}
      }
    }
  }

  void reap_idle() {
    std::vector<std::string> expired;
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) return;
      const auto now = Clock::now();
      for (auto& [sid, s] : sessions_)
        if (!s.busy && now - s.last_used >= seconds(settings_.session_idle_timeout)) {
          s.busy = true;
          expired.push_back(sid);
        }
    }
    for (const auto& sid : expired) close_allocation(sid);
  }

  void run() {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock q(queue_lock_);
        if (stopping_ && queue_.empty()) return;
      }
      try { reap_idle(); }
      catch (...) { fatal(); }
      {
        std::unique_lock q(queue_lock_);
        queue_cv_.wait_for(q, std::chrono::milliseconds(50), [&] { return !queue_.empty() || stopping_.load(); });
        if (queue_.empty()) continue;
        job = queue_.front();
        queue_.pop_front();
      }
      std::string result;
      try {
        if (failed()) throw ApiError(503, "worker_unavailable");
        if (Clock::now() >= job->deadline) throw ApiError(504, "queue_deadline");
        result = job->function();
      } catch (const ApiError&) {
        job->cleanup();
        job->promise.set_exception(std::current_exception());
        continue;
      } catch (...) {
        fatal();
        job->cleanup();
        job->promise.set_exception(std::make_exception_ptr(ApiError(503, "worker_unavailable")));
        continue;
      }
      // Publish completion and observe abandonment atomically. Only this owner thread may release an
      // abandoned native allocation.
      bool abandoned = false;
      {
        std::lock_guard g(job->completion);
        const bool expired = job->abandoned || Clock::now() >= job->deadline;
        abandoned = job->discard && expired;
        if (expired) {
          if (!abandoned) job->cleanup();
          job->promise.set_exception(std::make_exception_ptr(ApiError(504, "request_deadline_outcome_unknown")));
        } else {
          job->cleanup();
          job->promise.set_value(std::move(result));
        }
      }
      if (abandoned) {
        try { job->discard(); }
        catch (...) { fatal(); }
        job->cleanup();
      }
    }
  }

  Backend& backend_;
  const lfm3_text::Tokenizer& tokenizer_;
  const Policy policy_;
  Settings settings_;
  std::mutex lock_;
  std::map<std::string, Session> sessions_;
  bool failed_ = false, cleanup_failed_ = false;
  std::atomic<bool> stopping_{false};
  std::mutex frontend_lock_;
  std::mutex queue_lock_;
  std::condition_variable queue_cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::mutex active_lock_;
  std::map<std::string, Active> active_;
  bool backend_closed_ = false;  // guarded by active_lock_
  std::uint64_t sequence_ = 0;   // owner thread only
  std::atomic<bool> in_call_{false}, timed_out_{false};
  std::atomic<Clock::rep> call_started_{0};
  std::mutex done_lock_;
  std::condition_variable done_cv_;
  bool owner_done_ = false, watchdog_stop_ = false;  // guarded by done_lock_
  std::thread owner_, watchdog_;
};

// ------------------------------------------------------------------ HTTP

inline const char* reason_phrase(int status) {
  switch (status) {
    case 200: return "OK"; case 202: return "Accepted"; case 400: return "Bad Request"; case 401: return "Unauthorized";
    case 404: return "Not Found"; case 408: return "Request Timeout"; case 409: return "Conflict";
    case 413: return "Request Entity Too Large"; case 414: return "Request-URI Too Long"; case 415: return "Unsupported Media Type";
    case 429: return "Too Many Requests"; case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error"; case 501: return "Not Implemented"; case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout"; case 505: return "HTTP Version Not Supported";
    default: return "Unknown";
  }
}

inline std::string lower(std::string s) {
  for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return s;
}

struct Request {
  std::string method, path;
  std::vector<std::pair<std::string, std::string>> headers;  // lower-case name, raw value
  std::vector<std::string> all(const std::string& name) const {
    std::vector<std::string> v;
    for (const auto& [k, value] : headers) if (k == name) v.push_back(value);
    return v;
  }
};

// One accepted connection: HTTP/1.0, one request, then close (Handler with protocol_version HTTP/1.0).
class Connection {
 public:
  Connection(int fd, Service& service, std::string token) : fd_(fd), service_(service), token_(std::move(token)) {
    const double t = service.settings().socket_timeout;
    timeval tv{};
    tv.tv_sec = time_t(t);
    tv.tv_usec = suseconds_t((t - double(tv.tv_sec)) * 1e6);
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  }
  ~Connection() {
    ::shutdown(fd_, SHUT_WR);
    ::close(fd_);
  }

  void handle() {
    Request request;
    // The whole request line and header block must arrive within one socket timeout; a peer trickling
    // one byte per timeout must not hold a handler forever.
    const auto header_deadline = Clock::now() + seconds(service_.settings().socket_timeout);
    int status = 0;
    try {
      status = read_head(request, header_deadline);
    } catch (const std::exception&) {
      return;  // timeout or reset while reading headers: close without a reply
    }
    if (status == -1) return;
    if (status) { reply(status, error_body(status == 431 ? "header_limit" : status == 414 ? "request_line_limit"
                                           : status == 505 ? "http_version" : "bad_request")); return; }
    if (request.method == "GET") { get(request); return; }
    if (request.method == "POST") { post(request); return; }
    reply(501, error_body("unsupported_method"));
  }

 private:
  // One raw line as BufferedReader.readline(limit) returns it: up to and including '\n', at most
  // `limit` bytes, shorter at end of stream (empty: the stream ended).
  std::string read_line(size_t limit, Clock::time_point deadline) {
    std::string line;
    char c;
    while (line.size() < limit && read_byte(c, deadline)) {
      line += c;
      if (c == '\n') break;
    }
    return line;
  }

  // BaseHTTPRequestHandler.handle_one_request + parse_request, and http.client.parse_headers followed
  // by email.feedparser (compat32) header parsing. Returns 0 on success, -1 to close silently, or an
  // HTTP status to refuse with.
  int read_head(Request& r, Clock::time_point deadline) {
    const auto request_line = read_line(65537, deadline);
    if (request_line.empty()) return -1;
    if (request_line.size() > 65536) return 414;
    std::string line = request_line;
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    // str.split() on the latin-1 decoded line: ASCII whitespace, \x1c-\x1f, \x85 and \xa0.
    std::vector<std::string> words;
    {
      std::string w;
      for (const unsigned char c : line) {
        const bool space = c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f) || c == 0x85 || c == 0xa0;
        if (space) { if (!w.empty()) { words.push_back(w); w.clear(); } }
        else w += char(c);
      }
      if (!w.empty()) words.push_back(w);
    }
    if (words.empty()) return -1;
    if (words.size() != 3) return 400;  // two words is HTTP/0.9, which only http.server's GET path accepts
    const auto& version = words[2];
    if (version.rfind("HTTP/", 0) != 0) return 400;
    const auto number = version.substr(5);
    const auto dot = number.find('.');
    if (dot == std::string::npos || number.find('.', dot + 1) != std::string::npos) return 400;
    const auto major = number.substr(0, dot), minor = number.substr(dot + 1);
    for (const auto& part : {major, minor}) {
      if (part.empty() || part.size() > 10) return 400;
      for (const char c : part) if (c < '0' || c > '9') return 400;
    }
    if (std::stoll(major) >= 2) return 505;
    r.method = words[0];
    r.path = words[1];
    if (r.path.rfind("//", 0) == 0) {
      const auto first = r.path.find_first_not_of('/');
      r.path = "/" + (first == std::string::npos ? std::string() : r.path.substr(first));
    }
    // http.client.parse_headers: raw lines until a bare line ending or end of stream.
    std::string block;
    for (int count = 0;;) {
      const auto raw = read_line(65537, deadline);
      if (raw.size() > 65536) return 431;
      if (++count > 100) return 431;  // http.client counts the terminating line too: 99 headers at most
      if (raw.empty() || raw == "\r\n" || raw == "\n") break;
      block += raw;
    }
    // email.feedparser: logical lines split on \r\n, \r or \n (each keeps its ending).
    std::vector<std::string> lines;
    for (size_t i = 0; i < block.size();) {
      size_t j = i;
      while (j < block.size() && block[j] != '\r' && block[j] != '\n') ++j;
      if (j < block.size()) j += (block[j] == '\r' && j + 1 < block.size() && block[j + 1] == '\n') ? 2 : 1;
      lines.push_back(block.substr(i, j - i));
      i = j;
    }
    const auto header_start = [](const std::string& l) {
      // headerRE: '^(From |[\041-\071\073-\176]*:|[\t ])'
      if (l.rfind("From ", 0) == 0 || (!l.empty() && (l[0] == ' ' || l[0] == '\t'))) return true;
      for (const unsigned char c : l) {
        if (c == ':') return true;
        if (c < 0x21 || c > 0x7e) return false;
      }
      return false;
    };
    std::vector<std::vector<std::string>> fields;  // source lines of each header
    for (size_t i = 0; i < lines.size(); ++i) {
      const auto& l = lines[i];
      if (!header_start(l)) break;  // a blank or malformed line ends the headers; the rest is body
      if (l[0] == ' ' || l[0] == '\t') {
        if (!fields.empty()) fields.back().push_back(l);  // a leading continuation is dropped
        continue;
      }
      if (l.rfind("From ", 0) == 0) continue;  // unix-from line (or defect): not a header
      if (l[0] == ':') continue;               // missing header name: dropped, parsing continues
      fields.push_back({l});
    }
    for (const auto& source : fields) {
      const auto colon = source[0].find(':');
      std::string value = source[0].substr(colon + 1);
      value.erase(0, std::min(value.size(), value.find_first_not_of(" \t")));
      for (size_t i = 1; i < source.size(); ++i) value += source[i];
      while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) value.pop_back();
      r.headers.emplace_back(lower(source[0].substr(0, colon)), value);
    }
    return 0;
  }

  bool read_byte(char& c, Clock::time_point deadline) {
    if (pos_ == buffer_.size()) {
      buffer_.clear();
      pos_ = 0;
      if (!fill(deadline, 4096)) return false;
    }
    c = buffer_[pos_++];
    return true;
  }
  // Reads at least one byte before `deadline`; false on orderly EOF; throws on timeout or error.
  bool fill(Clock::time_point deadline, size_t want) {
    for (;;) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
      if (remaining <= 0) throw std::runtime_error("socket deadline");
      pollfd p{fd_, POLLIN, 0};
      const int ready = ::poll(&p, 1, int(std::min<long long>(remaining, 1 << 30)));
      if (ready < 0 && errno == EINTR) continue;
      if (ready <= 0) throw std::runtime_error("socket deadline");
      char block[65536];
      const auto n = ::recv(fd_, block, std::min(want, sizeof block), 0);
      if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      if (n < 0) throw std::runtime_error("socket error");
      if (n == 0) return false;
      buffer_.append(block, size_t(n));
      return true;
    }
  }

  void reply(int status, std::string body, size_t limit = kMaxResponse) {
    if (body.size() > limit) { status = 500; body = "{\"error\":\"response_limit\"}"; }
    char date[64];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    ::gmtime_r(&now, &tm);
    std::strftime(date, sizeof date, "%a, %d %b %Y %H:%M:%S GMT", &tm);
    std::string out = "HTTP/1.0 " + std::to_string(status) + " " + reason_phrase(status) + "\r\n";
    out += "Server: LFM3Local \r\nDate: ";  // http.server: server_version + " " + sys_version ("")
    out += date;
    out += "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: " + std::to_string(body.size()) +
           "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" + body;
    size_t sent = 0;
    while (sent < out.size()) {
      const auto n = ::send(fd_, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return;  // a disconnected or non-reading peer cannot receive a second reply
      sent += size_t(n);
    }
  }

  void get(const Request& r) {
    if (r.path != "/health") { reply(404, error_body("not_found")); return; }
    const bool ready = service_.ready();
    reply(ready ? 200 : 503, Obj().boolean("ready", ready).boolean("gpu_batching", false).done());
  }

  static bool constant_time_equal(const std::string& a, const std::string& b) {
    // hmac.compare_digest: time depends only on the length of `a` (the presented value).
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < a.size(); ++i) diff |= static_cast<unsigned char>(a[i] ^ (i < b.size() ? b[i] : 0));
    return diff == 0;
  }

  void post(const Request& r) {
    try {
      const auto auth = r.all("authorization");
      bool ascii = auth.size() == 1;
      if (ascii) for (const unsigned char c : auth[0]) if (c >= 0x80) ascii = false;
      if (auth.size() != 1 || auth[0].size() > 263 || !ascii || !constant_time_equal(auth[0], "Bearer " + token_))
        throw ApiError(401, "unauthorized");
      const auto lengths = r.all("content-length");
      if (!r.all("transfer-encoding").empty() || lengths.size() != 1) throw ApiError(400, "content_length_required");
      const auto& length_text = lengths[0];
      bool digits = !length_text.empty() && length_text.size() <= 8;
      for (const char c : length_text) if (c < '0' || c > '9') digits = false;
      if (!digits) throw ApiError(400, "invalid_content_length");
      const size_t length = std::stoul(length_text);
      if (length < 1 || length > kMaxBody) throw ApiError(413, "body_limit");
      const auto types = r.all("content-type");
      if (types.size() != 1 || lower(types[0]) != "application/json") throw ApiError(415, "application_json_required");
      const auto deadline = Clock::now() + seconds(service_.settings().socket_timeout);
      std::string body = buffer_.substr(pos_);
      while (body.size() < length) {
        if (Clock::now() >= deadline) throw ApiError(408, "body_deadline");
        buffer_.clear();
        pos_ = 0;
        try {
          if (!fill(deadline, std::min<size_t>(65536, length - body.size()))) throw ApiError(400, "incomplete_body");
        } catch (const ApiError&) {
          throw;
        } catch (const std::exception&) {
          throw ApiError(408, "socket_deadline");
        }
        body += buffer_;
      }
      body.resize(length);
      // A parsed body is a tree several times the size of its text. Bodies over 1 MiB are parsed one
      // at a time; with the try-only frontend lock this bounds the trees alive at once.
      std::shared_ptr<const Json> parsed;
      {
        static std::mutex large;
        std::unique_lock guard(large, std::defer_lock);
        if (length > (size_t(1) << 20)) guard.lock();
        parsed = std::make_shared<const Json>(strict_json(body));
      }
      std::string().swap(body);
      std::string().swap(buffer_);
      const Json& request = *parsed;
      std::string result;
      if (r.path == "/sessions") result = service_.create(request);
      else if (r.path == "/generate") result = service_.generate(std::move(parsed));
      else if (r.path == "/generate_batch") result = service_.generate_batch(std::move(parsed));
      else if (r.path == "/speech") { reply(200, service_.speech(std::move(parsed)), kMaxSpeechResponse); return; }
      else {
        // /sessions/<32 URL-safe characters>/(reset|close|cancel)
        const std::string prefix = "/sessions/";
        if (r.path.size() < prefix.size() + 33 || r.path.compare(0, prefix.size(), prefix) != 0 ||
            !valid_session_id(r.path.substr(prefix.size(), 32)) || r.path[prefix.size() + 32] != '/')
          throw ApiError(404, "not_found");
        const auto sid = r.path.substr(prefix.size(), 32);
        const auto op = r.path.substr(prefix.size() + 33);
        if (op == "cancel") { reply(202, service_.cancel(sid, request)); return; }
        if (op != "reset" && op != "close") throw ApiError(404, "not_found");
        result = service_.lifecycle(sid, op, request);
      }
      reply(200, result);
    } catch (const ApiError& e) {
      reply(e.status, error_body(e.code));
    } catch (const std::exception&) {
      reply(400, error_body("invalid_request"));
    }
  }

  int fd_;
  Service& service_;
  const std::string token_;
  std::string buffer_;
  size_t pos_ = 0;
};

// LocalHTTPServer: literal IPv4 loopback only, a bounded number of concurrent handlers, and on
// saturation no thread, parsing, authentication or admission work (the connection is closed).
class Server {
 public:
  Server(Service& service, std::string token, int port) : service_(service), token_(std::move(token)) {
    if (!valid_token(token_)) throw std::invalid_argument("bearer token must be 32..256 URL-safe characters");
    if (port < 0 || port > 65535) throw std::invalid_argument("port must be an integer in 0..65535");
    listener_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener_ < 0) throw std::runtime_error("socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(std::uint16_t(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 || ::listen(listener_, 5) != 0) {
      ::close(listener_);
      throw std::runtime_error(std::string("cannot listen on 127.0.0.1: ") + std::strerror(errno));
    }
    socklen_t size = sizeof address;
    ::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size);
    port_ = ntohs(address.sin_port);
    if (::pipe2(wake_, O_CLOEXEC | O_NONBLOCK) != 0) { ::close(listener_); throw std::runtime_error("pipe failed"); }
  }
  ~Server() {
    if (listener_ >= 0) ::close(listener_);
    ::close(wake_[0]);
    ::close(wake_[1]);
  }
  int port() const { return port_; }

  // Accepts until stop(). Running handlers continue; see drain().
  void serve() {
    for (;;) {
      pollfd p[2] = {{listener_, POLLIN, 0}, {wake_[0], POLLIN, 0}};
      if (::poll(p, 2, -1) < 0) { if (errno == EINTR) continue; break; }
      if (p[1].revents) break;
      if (!(p[0].revents & POLLIN)) continue;
      const int fd = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd < 0) continue;
      {
        std::lock_guard g(lock_);
        if (active_ >= service_.settings().handlers) { ::close(fd); continue; }
        ++active_;
      }
      try {
        std::thread([this, fd] {
          try { Connection(fd, service_, token_).handle(); } catch (...) {}
          std::lock_guard g(lock_);
          --active_;
          idle_.notify_all();
        }).detach();
      } catch (...) {
        ::close(fd);
        std::lock_guard g(lock_);
        --active_;
      }
    }
    // Admission is over: refuse new connections at once rather than leaving them in the backlog.
    ::close(listener_);
    listener_ = -1;
  }
  // Waits up to `limit` for running handlers to send their replies.
  void drain(std::chrono::nanoseconds limit) {
    std::unique_lock g(lock_);
    idle_.wait_for(g, limit, [&] { return active_ == 0; });
  }
  // Async-signal-safe.
  void stop() {
    const char c = 1;
    [[maybe_unused]] const auto n = ::write(wake_[1], &c, 1);
  }
  int active_handlers() {
    std::lock_guard g(lock_);
    return active_;
  }

 private:
  Service& service_;
  std::string token_;
  int listener_ = -1, port_ = 0;
  int wake_[2] = {-1, -1};
  std::mutex lock_;
  std::condition_variable idle_;
  int active_ = 0;
};

}  // namespace lfm::http
