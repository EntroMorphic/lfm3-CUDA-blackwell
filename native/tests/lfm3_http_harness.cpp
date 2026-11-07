// Host harness for lfm/lfm3_http.hpp: the real HTTP layer, Service, native tokenizer and session ledger,
// with a fake engine in place of the GPU. The fake speaks resident protocol 2 exactly as
// liquid-lfm3-worker does (same reply fields, same cancellation pipe reader), so every Service path,
// including deadlines, abandonment, cancellation, batching and fatal failure, runs on any host.
//
//   lfm3-http-harness CHECKPOINT_DIR [--mode phrase|stop] [--token-delay-ms N] [--create-delay-ms N]
//                     [--fail-op OP] [--fail-after N] [--port N] [--request-timeout S] [--socket-timeout S]
//                     [--session-idle-timeout S] [--total-capacity N] [--max-sessions N] [--queue-size N]
//                     [--handlers N]
// Token from LFM3_SERVER_TOKEN. Prints {"ready":true,"port":N}. SIGTERM exits like liquid-lfm3-server.
#include <lfm/lfm3_http.hpp>
#include <lfm/lfm3_text_request.hpp>
#include <lfm/text_batch_cancellation.hpp>
#include <lfm/text_cancellation.hpp>
#include <lfm/text_sessions.hpp>

#include <csignal>
#include <iostream>
#include <pthread.h>
#include <sstream>

namespace {

using lfm::Json;
using lfm::http::Obj;

// Generation content for --mode phrase: exercises non-ASCII, JSON escapes and a final stop.
const char* kPhrase = "Caf\xc3\xa9 \"quoted\" back\\slash\ttab\nline \xe2\x80\xa8 \x7f end \xf0\x9f\x99\x82 done.";

class FakeEngine final : public lfm::http::Backend {
 public:
  FakeEngine(const lfm::lfm3_text::Tokenizer& t, std::vector<int> stops, std::string mode, int token_delay_ms,
             int create_delay_ms, std::string fail_op, int fail_after)
      : tokenizer_(t), stops_(std::move(stops)), mode_(std::move(mode)), token_delay_ms_(token_delay_ms),
        create_delay_ms_(create_delay_ms), fail_op_(std::move(fail_op)), fail_after_(fail_after) {
    if (::pipe2(pipe_, O_NONBLOCK | O_CLOEXEC) != 0) throw std::runtime_error("pipe");
    reader_ = std::make_unique<lfm::text::CancellationReader>(pipe_[0]);
    phrase_ = tokenizer_.encode(kPhrase);
  }
  int cancellation_fd() const override { return pipe_[1]; }
  void close() override { closed_ = true; }

  std::string call(const std::string& line) override {
    if (closed_) throw std::runtime_error("engine closed");
    std::ostringstream out;
    int rpc = 0;
    try {
      const auto request = Json::parse(line);
      rpc = request.at("rpc_id").integer();
      if (rpc != last_rpc_ + 1) throw std::runtime_error("out-of-sequence RPC ID");
      last_rpc_ = rpc;
      const auto op = request.at("op").string();
      if (op == fail_op_ && ++fail_count_ >= fail_after_) throw std::runtime_error("injected failure");
      if (op == "generate_batch") { batch(request, rpc, out); return out.str(); }
      const auto id = request.at("session_id").string();
      if (op == "create") {
        std::this_thread::sleep_for(std::chrono::milliseconds(create_delay_ms_));
        ledger_.create(id, request.at("capacity").integer());
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"session_id\":\"" << id << "\"}\n";
      } else if (op == "close") {
        ledger_.close(id);
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"closed\":true}\n";
      } else if (op == "reset") {
        ledger_.reset(id);
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"reset\":true}\n";
      } else if (op == "generate") {
        Lane lane = start(request);
        for (;;) {
          if (reader_->requested(rpc)) { lane.lease->abort(); reply(out, rpc, lane, true, -1); return out.str(); }
          if (step(lane)) break;
        }
        finish(lane);
        reply(out, rpc, lane, false, -1);
      } else throw std::runtime_error("unknown operation");
    } catch (const std::exception& e) {
      out.str("");
      out << "{\"rpc_id\":" << rpc << ",\"ok\":false,\"error\":";
      std::string message;
      lfm::http::append_py_string(message, e.what());
      out << message << "}\n";
    }
    return out.str();
  }

 private:
  struct Lane {
    std::string id;
    std::vector<int> input, generated;
    int count = 0;
    bool do_sample = false;
    long long seed = 0;
    size_t reused = 0;
    std::unique_ptr<lfm::text::SessionLedger::Lease> lease;
    bool done = false, cancelled = false;
  };
  Lane start(const Json& r) {
    Lane lane;
    lane.id = r.at("session_id").string();
    for (const auto& v : r.at("input_ids").array()) lane.input.push_back(v.integer());
    lane.count = r.at("max_new_tokens").integer();
    lane.do_sample = r.at("do_sample").boolean();
    lane.seed = r.at("seed").integer();
    if (!r.at("return_text").boolean()) throw std::runtime_error("server must ask for text");
    lane.lease = std::make_unique<lfm::text::SessionLedger::Lease>(
        ledger_.begin(lane.id, lfm::text::SessionLedger::Tokens(lane.input.begin(), lane.input.end()), lane.count));
    lane.reused = lane.lease->plan().full_prefill ? 0 : lane.lease->plan().reuse_tokens;
    return lane;
  }
  // Emits one token; true when the lane is complete.
  bool step(Lane& lane) {
    std::this_thread::sleep_for(std::chrono::milliseconds(token_delay_ms_));
    const size_t i = lane.generated.size();
    int token = mode_ == "stop" ? stops_.front() : i < phrase_.size() ? phrase_[i] : stops_.front();
    lane.generated.push_back(token);
    return std::find(stops_.begin(), stops_.end(), token) != stops_.end() || lane.generated.size() == size_t(lane.count);
  }
  void finish(Lane& lane) {
    std::vector<int> consumed = lane.input;
    consumed.insert(consumed.end(), lane.generated.begin(), lane.generated.end() - 1);
    lane.lease->commit(lfm::text::SessionLedger::Tokens(lane.generated.begin(), lane.generated.end()),
                       lfm::text::SessionLedger::Tokens(consumed.begin(), consumed.end()));
    lane.done = true;
  }
  void reply(std::ostream& out, int rpc, const Lane& lane, bool cancelled, long long paired) {
    std::string text;
    lfm::http::append_py_string(text, lfm::lfm3_text::generated_text(tokenizer_, lane.generated, stops_));
    std::string ids = "[";
    for (size_t i = 0; i < lane.generated.size(); ++i) ids += (i ? "," : "") + std::to_string(lane.generated[i]);
    ids += "]";
    const size_t consumed = cancelled ? 0 : lane.input.size() + lane.generated.size() - 1;
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"session_id\":\"" << lane.id << "\",\"generated_ids\":" << ids
        << ",\"reused_tokens\":" << (cancelled ? 0 : lane.reused) << ",\"consumed_tokens\":" << consumed
        << ",\"do_sample\":" << (lane.do_sample ? "true" : "false") << ",\"seed\":" << lane.seed;
    if (cancelled || paired >= 0) out << ",\"cancelled\":" << (cancelled ? "true" : "false") << ",\"cache_valid\":" << (cancelled ? "false" : "true");
    if (paired >= 0) out << ",\"gpu_batching\":" << (paired ? "true" : "false") << ",\"paired_decode_calls\":" << paired;
    out << ",\"live_token_count\":125033,\"text\":" << text << "}";
  }
  void batch(const Json& request, int rpc, std::ostream& out) {
    const auto& lanes_json = request.at("lanes").array();
    std::array<Lane, 2> lanes{start(lanes_json[0]), start(lanes_json[1])};
    long long paired = 0;
    for (;;) {
      const auto c = reader_->requested_pair(rpc);
      for (size_t i = 0; i < 2; ++i)
        if (!lanes[i].done && c[i]) { lanes[i].lease->abort(); lanes[i].cancelled = lanes[i].done = true; }
      if (lanes[0].done && lanes[1].done) break;
      int stepping = 0;
      for (auto& lane : lanes)
        if (!lane.done) { ++stepping; if (step(lane)) finish(lane); }
      // Two lanes that both still need a forward after this token share one paired call.
      if (stepping == 2 && !lanes[0].done && !lanes[1].done) ++paired;
    }
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"gpu_batching\":" << (paired ? "true" : "false")
        << ",\"paired_decode_calls\":" << paired << ",\"results\":[";
    for (size_t i = 0; i < 2; ++i) {
      if (i) out << ',';
      reply(out, rpc, lanes[i], lanes[i].cancelled, paired);
    }
    out << "]}\n";
  }

  const lfm::lfm3_text::Tokenizer& tokenizer_;
  std::vector<int> stops_, phrase_;
  std::string mode_;
  int token_delay_ms_, create_delay_ms_;
  std::string fail_op_;
  int fail_after_, fail_count_ = 0;
  int pipe_[2] = {-1, -1};
  std::unique_ptr<lfm::text::CancellationReader> reader_;
  lfm::text::SessionLedger ledger_{2, 32768};
  int last_rpc_ = 0;
  bool closed_ = false;
};

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2 || argc % 2 != 0) throw std::runtime_error("usage: lfm3-http-harness CHECKPOINT_DIR [--option value]...");
  const std::string directory = argv[1];
  lfm::http::Settings settings;
  std::string mode = "phrase", fail_op;
  int port = 0, token_delay = 0, create_delay = 0, fail_after = 1;
  for (int i = 2; i < argc; i += 2) {
    const std::string o = argv[i], v = argv[i + 1];
    if (o == "--mode") mode = v;
    else if (o == "--token-delay-ms") token_delay = std::stoi(v);
    else if (o == "--create-delay-ms") create_delay = std::stoi(v);
    else if (o == "--fail-op") fail_op = v;
    else if (o == "--fail-after") fail_after = std::stoi(v);
    else if (o == "--port") port = std::stoi(v);
    else if (o == "--request-timeout") settings.request_timeout = std::stod(v);
    else if (o == "--socket-timeout") settings.socket_timeout = std::stod(v);
    else if (o == "--session-idle-timeout") settings.session_idle_timeout = std::stod(v);
    else if (o == "--total-capacity") settings.total_capacity = std::stoi(v);
    else if (o == "--max-sessions") settings.max_sessions = std::stoi(v);
    else if (o == "--queue-size") settings.queue_size = std::stoi(v);
    else if (o == "--handlers") settings.handlers = std::stoi(v);
    else if (o == "--engine-timeout") settings.engine_timeout = std::stod(v);
    else throw std::runtime_error("unsupported option " + o);
  }
  const char* token = std::getenv("LFM3_SERVER_TOKEN");
  if (!token) throw std::runtime_error("LFM3_SERVER_TOKEN required");
  sigset_t signals;
  sigemptyset(&signals); sigaddset(&signals, SIGTERM); sigaddset(&signals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  std::signal(SIGPIPE, SIG_IGN);

  const auto config = lfm::ModelConfig::load(directory + "/config.json", 32768);
  std::ifstream raw_config(directory + "/config.json");
  const auto model = Json::parse(std::string(std::istreambuf_iterator<char>(raw_config), {}));
  lfm::http::Policy policy;
  policy.vocabulary = config.shape.vocabulary;
  policy.stops = config.eos;
  for (auto name : {"audio_token_id", "audio_out_token_id", "image_token_id", "video_token_id", "mixed_start_token_id", "audio_start_token_id"})
    policy.suppress.push_back(model.at(name).integer());
  policy.stops.push_back(model.at("text_end_token_id").integer());
  for (auto* ids : {&policy.suppress, &policy.stops}) {
    std::sort(ids->begin(), ids->end());
    ids->erase(std::unique(ids->begin(), ids->end()), ids->end());
  }
  const auto tokenizer = lfm::lfm3_text::Tokenizer::load(directory + "/tokenizer.json");
  FakeEngine engine(tokenizer, policy.stops, mode, token_delay, create_delay, fail_op, fail_after);
  auto service = std::make_unique<lfm::http::Service>(engine, tokenizer, policy, settings);
  auto server = std::make_unique<lfm::http::Server>(*service, token, port);
  std::thread waiter([&] {
    int received = 0;
    sigwait(&signals, &received);
    server->stop();
    sigwait(&signals, &received);
    std::_Exit(1);
  });
  std::cout << "{\"ready\":true,\"port\":" << server->port() << "}\n" << std::flush;
  server->serve();
  service->close();
  server->drain(lfm::http::seconds(settings.socket_timeout));
  std::cout << std::flush;
  std::_Exit(service->cleanup_failed() ? 1 : 0);
} catch (const std::exception& e) {
  std::cerr << "lfm3-http-harness: " << e.what() << "\n";
  return 1;
}
