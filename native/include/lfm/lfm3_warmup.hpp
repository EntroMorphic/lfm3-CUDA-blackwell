#pragma once
// validation/always-on-v1: liquid-lfm3-server --warmup. Before the ready line, one request of each kind (text /generate
// in its own session, TTS /speech streamed, a 4K image, audio in) goes through the same Service a client's request
// goes through (validation, the front end, the queue, the engine) and its reply is discarded. Meanwhile the HTTP
// server already accepts: /health answers 503 and every POST 503 warming_up (Service::set_warming). The inputs are
// generated here, deterministically, and come from no test set: a 3840 x 2160 PNG of coloured tiles, a 2 s 16 kHz
// tone glide, fixed short text.
//
// Rule 3 (nothing left behind): the text session is closed; the queue is checked empty; and the warm-up image is never
// cached on either side: the front end's cache budget is 0 while warming (Service::set_warming) and the engine's is
// set to 0 by the caller before the engine starts and restored here on the engine-owner thread afterwards
// (`engine_image_cache`). Then both caches, the engine's published keys (lfm::image::shared_held_images) and the pixel
// handoff are checked empty. Skipping rather than clearing: nothing to undo, no new engine operation, and the
// generated image (which any client could rebuild from this file) is never in a cache, not even for a moment.
//
// serve_with_warmup runs it all for both liquid-lfm3-server and the host harness. A failure prints why and exits 1;
// the server never serves cold without saying so. Host-only (no CUDA).
#include "lfm/lfm3_http.hpp"

#include <sys/un.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>

namespace lfm::http::warmup {

// ------------------------------------------------------------------ generated inputs

namespace detail {

inline std::uint32_t crc32(const std::string& data, size_t from = 0) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
      t[n] = c;
    }
    return t;
  }();
  std::uint32_t c = 0xffffffffu;
  for (size_t i = from; i < data.size(); ++i) c = table[(c ^ std::uint8_t(data[i])) & 0xff] ^ (c >> 8);
  return c ^ 0xffffffffu;
}

inline std::uint32_t adler32(const std::string& data) {
  std::uint32_t a = 1, b = 0;
  for (const char c : data) {
    a = (a + std::uint8_t(c)) % 65521;
    b = (b + a) % 65521;
  }
  return (b << 16) | a;
}

inline void be32(std::string& out, std::uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) out += char((v >> s) & 0xff);
}

// RFC 1951 with the fixed Huffman codes, one final block: literals and matches at distance 1 or 3 (runs of a byte,
// runs of a pixel). Enough for an image of flat tiles; any inflater reads it.
class Deflate {
 public:
  std::string compress(const std::string& in) {
    bits(1, 1);  // BFINAL
    bits(1, 2);  // BTYPE 01: fixed codes
    size_t i = 0;
    while (i < in.size()) {
      size_t best = 0, distance = 0;
      for (const size_t d : {size_t(1), size_t(3)}) {
        if (i < d) continue;
        size_t n = 0;
        while (n < 258 && i + n < in.size() && in[i + n] == in[i + n - d]) ++n;
        if (n > best) best = n, distance = d;
      }
      if (best >= 3) {
        length(best);
        code(distance == 1 ? 0 : 2, 5);  // distance codes 0 (1) and 2 (3): 5 bits, no extra bits
        i += best;
      } else {
        literal(std::uint8_t(in[i++]));
      }
    }
    literal(256);  // end of block
    if (used_) out_ += char(acc_);
    return out_;
  }

 private:
  void bits(std::uint32_t v, int n) {  // LSB first
    for (int k = 0; k < n; ++k) {
      acc_ |= ((v >> k) & 1u) << used_;
      if (++used_ == 8) out_ += char(acc_), acc_ = 0, used_ = 0;
    }
  }
  void code(std::uint32_t c, int n) {  // a Huffman code, most significant bit first
    for (int k = n - 1; k >= 0; --k) bits((c >> k) & 1u, 1);
  }
  void literal(int v) {
    if (v < 144) code(0x30 + v, 8);
    else if (v < 256) code(0x190 + (v - 144), 9);
    else if (v < 280) code(v - 256, 7);
    else code(0xc0 + (v - 280), 8);
  }
  void length(size_t n) {
    static const int base[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
                                 115, 131, 163, 195, 227, 258};
    static const int extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    int k = 28;
    while (base[k] > int(n)) --k;
    literal(257 + k);
    bits(std::uint32_t(int(n) - base[k]), extra[k]);
  }
  std::string out_;
  std::uint32_t acc_ = 0;
  int used_ = 0;
};

}  // namespace detail

inline constexpr int kImageWidth = 3840, kImageHeight = 2160;

// The warm-up image: 3840 x 2160 RGB, 8-bit, tiles of 120 x 120 pixels whose colours come from an integer hash of the
// tile's place. PNG with each tile band's first row Sub-filtered and the rest Up-filtered (all zeros), so it deflates
// to a few hundred KB. The same bytes on every machine.
inline std::string png_4k() {
  constexpr int w = kImageWidth, h = kImageHeight, tile = 120;
  const auto colour = [](int tx, int ty, int c) {
    std::uint32_t x = std::uint32_t(tx) * 73856093u ^ std::uint32_t(ty) * 19349663u ^ std::uint32_t(c) * 83492791u;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    return char(x & 0xff);
  };
  std::string raw;
  raw.reserve(size_t(h) * (1 + size_t(w) * 3));
  for (int y = 0; y < h; ++y) {
    if (y % tile) {  // filter Up: the same row as the one above
      raw += char(2);
      raw.append(size_t(w) * 3, '\0');
      continue;
    }
    raw += char(1);  // filter Sub: each byte less the byte one pixel before
    std::string row(size_t(w) * 3, '\0');
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) row[size_t(x) * 3 + size_t(c)] = colour(x / tile, y / tile, c);
    for (size_t i = row.size(); i-- > 3;) row[i] = char(std::uint8_t(row[i]) - std::uint8_t(row[i - 3]));
    raw += row;
  }
  std::string z = "\x78\x01";
  z += detail::Deflate().compress(raw);
  detail::be32(z, detail::adler32(raw));
  std::string png = "\x89PNG\r\n\x1a\n";
  const auto chunk = [&](const char* type, const std::string& data) {
    detail::be32(png, std::uint32_t(data.size()));
    const size_t from = png.size();
    png += type;
    png += data;
    detail::be32(png, detail::crc32(png, from));
  };
  std::string ihdr;
  detail::be32(ihdr, w);
  detail::be32(ihdr, h);
  ihdr += std::string("\x08\x02\x00\x00\x00", 5);  // 8-bit RGB, deflate, adaptive filtering, no interlace
  chunk("IHDR", ihdr);
  chunk("IDAT", z);
  chunk("IEND", "");
  return png;
}

// The warm-up audio: 2 s of 16 kHz mono 16-bit PCM WAV, a triangle wave gliding from 220 Hz to 880 Hz under a rise and
// fall, integer arithmetic only (the same bytes on every machine).
inline std::string wav_clip() {
  constexpr std::uint32_t rate = 16000, n = 2 * rate;
  std::string pcm;
  pcm.reserve(n * 2);
  std::uint32_t phase = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t hz = 220 + 660 * i / n;
    phase += hz * 65536u / rate;  // 16.16 cycles
    const std::int32_t p = std::int32_t(phase & 0xffff);  // position in the cycle, 0..65535
    const std::int32_t tri = p < 32768 ? p - 16384 : 49152 - p;  // -16384..16384
    const std::int32_t env = std::int32_t(i < n / 2 ? i : n - i) * 2;  // 0..n
    const std::int32_t s = std::int32_t(std::int64_t(tri) * env / std::int32_t(n) / 2);  // at most 8192
    pcm += char(s & 0xff);
    pcm += char((s >> 8) & 0xff);
  }
  const auto le = [](std::string& o, std::uint32_t v, int bytes) {
    for (int k = 0; k < bytes; ++k) o += char((v >> (8 * k)) & 0xff);
  };
  std::string wav = "RIFF";
  le(wav, 36 + n * 2, 4);
  wav += "WAVEfmt ";
  le(wav, 16, 4);
  le(wav, 1, 2);         // PCM
  le(wav, 1, 2);         // mono
  le(wav, rate, 4);
  le(wav, rate * 2, 4);  // byte rate
  le(wav, 2, 2);         // block align
  le(wav, 16, 2);        // bits
  wav += "data";
  le(wav, n * 2, 4);
  return wav + pcm;
}

// The four request bodies, as a client would send them.
inline std::string text_body(const std::string& sid) {
  return Obj().str("session_id", sid)
      .raw("messages", "[{\"role\":\"user\",\"content\":\"Warm-up check: reply with one short greeting.\"}]")
      .num("max_new_tokens", 16).done();
}
inline std::string tts_body() {
  return Obj()
      .raw("messages", "[{\"role\":\"system\",\"content\":\"Perform TTS in American English using a female voice.\"},"
                       "{\"role\":\"user\",\"content\":\"The service is warming up.\"}]")
      .str("mode", "audio").num("max_new_tokens", 48).num("seed", 0).boolean("stream", true).done();
}
inline std::string image_body() {
  std::string messages = "[{\"role\":\"system\",\"content\":\"Describe the provided image.\"},{\"role\":\"user\","
                         "\"content\":[{\"type\":\"image\",\"image\":\"";
  messages += lfm::base64::encode(png_4k());
  messages += "\"},{\"type\":\"text\",\"text\":\"What is in the image?\"}]}]";
  return Obj().raw("messages", messages).str("mode", "text").num("max_new_tokens", 8).num("seed", 0).done();
}
inline std::string audio_body() {
  std::string messages = "[{\"role\":\"system\",\"content\":\"Respond with interleaved text and audio.\"},{\"role\":"
                         "\"user\",\"content\":[{\"type\":\"audio\",\"audio\":\"";
  messages += lfm::base64::encode(wav_clip());
  messages += "\"}]}]";
  return Obj().raw("messages", messages).str("mode", "interleaved").num("max_new_tokens", 32).num("seed", 0).done();
}

// ------------------------------------------------------------------ the warm-up

struct Step {
  std::string kind;
  double seconds = 0;
  std::string skipped;                      // why the kind was not run (the engine cannot), or empty
  std::optional<double> prefill, generate;  // the engine's own timings, from /speech replies
};
struct Report {
  std::vector<Step> steps;
  double seconds = 0;  // the whole warm-up
};

inline double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Runs the four requests on `service` (set_warming(true) already, the engine's image cache budget already 0), checks
// rule 3, restores the engine's image cache through `engine_image_cache` (called on the engine-owner thread) and ends
// the warming (set_warming(false)). Throws std::runtime_error naming the request and the reason on any failure.
inline Report run(Service& service, const std::function<void()>& engine_image_cache) {
  Report report;
  const auto t_all = Clock::now();
  const auto parse = [](const std::string& s) { return std::make_shared<const Json>(strict_json(s)); };
  const auto fail = [](const std::string& kind, const std::string& why) -> void {
    throw std::runtime_error("warm-up " + kind + " request: " + why);
  };
  const auto attempt = [&](const std::string& kind, const std::function<void(Step&)>& body) {
    Step step;
    step.kind = kind;
    const auto t0 = Clock::now();
    try {
      body(step);
    } catch (const ApiError& e) {
      fail(kind, std::to_string(e.status) + " " + e.code);
    } catch (const std::exception& e) {
      fail(kind, e.what());
    }
    step.seconds = since(t0);
    report.steps.push_back(std::move(step));
  };
  const auto timings = [](Step& step, const Json& reply) {
    for (const auto& [key, out] : {std::pair{"prefill_seconds", &step.prefill}, {"generate_seconds", &step.generate}})
      if (reply.has(key))
        if (const auto* v = std::get_if<double>(&reply.at(key).value)) *out = *v;
  };
  const auto speech_reply = [&](Step& step, const std::string& reply) {
    const Json r = strict_json(reply);
    if (!r.has("cancelled") || r.at("cancelled").boolean()) throw std::runtime_error("not completed: " + reply.substr(0, 200));
    timings(step, r);
  };
  const auto& policy = service.policy();
  const bool speech = policy.speech && service.settings().engine_timeout >= kSpeechMinEngineTimeout;
  const auto skip = [&](const std::string& kind, const std::string& why) {
    Step step;
    step.kind = kind;
    step.skipped = why;
    report.steps.push_back(std::move(step));
  };

  attempt("text", [&](Step&) {
    const auto created = strict_json(service.create(strict_json("{\"capacity\":1024}")));
    const std::string sid = created.at("session_id").string();
    std::string reply;
    try {
      reply = service.generate(parse(text_body(sid)));
    } catch (...) {
      try { service.lifecycle(sid, "close", strict_json("{}")); } catch (...) {}
      throw;
    }
    service.lifecycle(sid, "close", strict_json("{}"));
    if (!strict_json(reply).has("text")) throw std::runtime_error("no text in the reply");
  });
  if (!speech) {
    skip("tts", "speech_unsupported");
  } else {
    attempt("tts", [&](Step& step) {
      std::string last;
      bool audio = false;
      Service::StreamSink sink;
      sink.send = [&](const std::string& line) {
        audio |= line.rfind("{\"event\":\"audio\"", 0) == 0;
        last = line;
        return true;
      };
      sink.commit = [] { return true; };
      sink.peer = [] { return Service::Peer::Open; };
      service.speech_stream(parse(tts_body()), sink);
      const Json done = strict_json(last.empty() ? "{}" : last);
      if (!done.has("event") || done.at("event").string() != "done") throw std::runtime_error("stream ended: " + last.substr(0, 200));
      if (!audio) throw std::runtime_error("no audio event");
      speech_reply(step, last);
    });
  }
  if (!speech || !policy.image_input) skip("image", "image_input_unsupported");
  else attempt("image", [&](Step& step) { speech_reply(step, service.speech(parse(image_body()))); });
  if (!speech || !policy.audio_input) skip("audio", "audio_input_unsupported");
  else attempt("audio", [&](Step& step) { speech_reply(step, service.speech(parse(audio_body()))); });

  // rule 3: nothing left behind (every request above has had its reply: its session closed, its queue entry gone, its
  // image leases released)
  const auto check = [&](bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("warm-up left " + what);
  };
  check(service.session_count() == 0, "a session");
  check(service.queued() == 0, "a request queued");
  check(service.image_cache_entries() == 0, "an entry in the front end's image cache");
  check(lfm::image::shared_held_images().size() == 0, "an entry in the engine's image cache");
  check(lfm::image::shared_pixel_handoff().size() == 0, "an image in the pixel handoff");
  if (engine_image_cache) {
    try {
      service.engine_task(engine_image_cache);
    } catch (const ApiError& e) {
      fail("engine image cache", std::to_string(e.status) + " " + e.code);
    }
  }
  if (service.failed()) throw std::runtime_error("the engine failed during warm-up");
  service.set_warming(false);
  report.seconds = since(t_all);
  return report;
}

// Seconds since this process started (/proc/self/stat's start time against CLOCK_BOOTTIME: the kernel's own clock for
// it), or nullopt where /proc cannot tell.
inline std::optional<double> process_age() {
  std::ifstream f("/proc/self/stat");
  std::string stat((std::istreambuf_iterator<char>(f)), {});
  const auto close = stat.rfind(')');
  if (close == std::string::npos) return std::nullopt;
  std::istringstream fields(stat.substr(close + 2));
  std::string field;
  for (int k = 3; k <= 22 && fields >> field; ++k)  // fields 3.. after "pid (comm)"; 22 is starttime
    if (k == 22) {
      timespec now{};
      const long ticks = ::sysconf(_SC_CLK_TCK);
      if (ticks <= 0 || ::clock_gettime(CLOCK_BOOTTIME, &now) != 0) return std::nullopt;
      const double started = std::stod(field) / double(ticks);
      return double(now.tv_sec) + double(now.tv_nsec) * 1e-9 - started;
    }
  return std::nullopt;
}

inline std::string fixed(double v, int digits = 3) {
  char b[64];
  std::snprintf(b, sizeof b, "%.*f", digits, v);
  return b;
}

// The one stderr line: {"warmup":{"startup_seconds", "warmup_seconds", "requests": [{"kind", "seconds" (or
// "skipped"), "prefill_seconds", "generate_seconds" (the engine's, /speech only)}]}}
inline std::string report_json(const Report& r, std::optional<double> startup) {
  std::string steps = "[";
  for (const auto& s : r.steps) {
    Obj o;
    o.str("kind", s.kind);
    if (!s.skipped.empty()) o.str("skipped", s.skipped);
    else o.raw("seconds", fixed(s.seconds));
    if (s.prefill) o.raw("prefill_seconds", fixed(*s.prefill, 4));
    if (s.generate) o.raw("generate_seconds", fixed(*s.generate, 4));
    steps += (steps.size() > 1 ? "," : "") + o.done();
  }
  steps += "]";
  Obj w;
  if (startup) w.raw("startup_seconds", fixed(*startup));
  w.raw("warmup_seconds", fixed(r.seconds)).raw("requests", steps);
  return Obj().raw("warmup", w.done()).done();
}

// sd_notify(3)'s READY=1, without libsystemd: one datagram to $NOTIFY_SOCKET (a path, or "@" for the abstract
// namespace) when systemd set it (a Type=notify unit); nothing otherwise. Best effort: false if it could not be sent.
inline bool notify_ready() {
  const char* path = std::getenv("NOTIFY_SOCKET");
  if (!path || (path[0] != '/' && path[0] != '@')) return false;
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  const size_t n = std::strlen(path);
  if (n < 2 || n >= sizeof a.sun_path) return false;
  std::memcpy(a.sun_path, path, n);
  if (a.sun_path[0] == '@') a.sun_path[0] = '\0';
  const int fd = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return false;
  static constexpr char kReady[] = "READY=1";
  const auto sent = ::sendto(fd, kReady, sizeof kReady - 1, MSG_NOSIGNAL, reinterpret_cast<const sockaddr*>(&a),
                             socklen_t(offsetof(sockaddr_un, sun_path) + n));
  ::close(fd);
  return sent == ssize_t(sizeof kReady - 1);
}

// The --warmup start, after the Service and Server exist and before anything was printed: serve() on this thread at
// once (so /health answers 503 while warming), the warm-up on another; once it ends, the stderr report, then the
// ready line {"ready":true,"port":N,"warmup_seconds":S} and READY=1. Returns the process's exit status: as without
// --warmup (0, or 1 when shutdown cleanup failed) after a stop; 1 after a warm-up failure, its reason on stderr.
// `stopped` is set by the caller's signal thread before it stops the server: a stop during warm-up is a stop (exit 0,
// "stopped during warm-up" on stderr), not a failure. `name` prefixes the stderr lines.
inline int serve_with_warmup(Service& service, Server& server, const std::function<void()>& engine_image_cache,
                             const std::atomic<bool>& stopped, const std::string& name) {
  service.set_warming(true);
  std::string failure;
  std::thread warm([&] {
    try {
      const auto report = run(service, engine_image_cache);
      if (stopped) throw std::runtime_error("stopped");
      std::cerr << report_json(report, process_age()) << "\n" << std::flush;
      std::cout << "{\"ready\":true,\"port\":" << server.port() << ",\"warmup_seconds\":" << fixed(report.seconds)
                << "}\n" << std::flush;
      notify_ready();
    } catch (const std::exception& e) {
      failure = e.what();
      server.stop();
    }
  });
  server.serve();
  service.close();
  warm.join();
  server.drain(seconds(service.settings().socket_timeout));
  std::cout << std::flush;
  const int status = service.cleanup_failed() ? 1 : 0;
  if (!failure.empty()) {
    if (stopped) {
      std::cerr << name << ": stopped during warm-up\n" << std::flush;
      return status;
    }
    std::cerr << name << ": warm-up failed: " << failure << "\n" << std::flush;
    return 1;
  }
  return status;
}

}  // namespace lfm::http::warmup
