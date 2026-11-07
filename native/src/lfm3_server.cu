// liquid-lfm3-server: one executable for LFM3 text serving. Loopback HTTP (lfm/lfm3_http.hpp, a port of
// tools/lfm3_server.py), the native text frontend (lfm/lfm3_text.hpp) and the engine (WorkerCore, the
// resident protocol of liquid-lfm3-worker, driven in-process by one engine-owner thread).
//
//   liquid-lfm3-server --experimental MODEL_DIRECTORY [--token-file FILE] [--port N] [--request-timeout S]
//                      [--socket-timeout S] [--session-idle-timeout S] [--engine-timeout S] [--total-capacity N]
//                      [--max-sessions N] [--queue-size N] [--handlers N]
// The bearer token comes from --token-file (an owned mode-0600 file) or LFM3_SERVER_TOKEN. --port 0
// picks a free port. Once listening, one line {"ready":true,"port":N} is written to stdout. SIGTERM or
// SIGINT stops admission, cancels running generations at their next token boundary, and exits 0; a
// second signal exits at once (status 1). An engine call longer than --engine-timeout (default 300 s,
// the Python worker client's) fails the server closed: every request then gets 503; restart it.
#define LFM3_WORKER_CORE_ONLY
#include "lfm3_worker.cu"
#include <lfm/lfm3_http.hpp>

#include <csignal>
#include <pthread.h>

namespace {

class CoreBackend final : public lfm::http::Backend {
 public:
  explicit CoreBackend(const std::string& directory) {
    if(::pipe2(pipe_,O_NONBLOCK|O_CLOEXEC)!=0) throw std::runtime_error("cancellation pipe failed");
    core_=std::make_unique<WorkerCore>(directory,"",false,pipe_[0]);
  }
  WorkerCore& core() {return *core_;}
  std::string call(const std::string& line) override {
    if(closed_) throw std::runtime_error("engine closed");
    std::ostringstream out;
    if(!core_->handle(line,false,out)) throw std::runtime_error("engine shut down");
    return out.str();
  }
  int cancellation_fd() const override {return pipe_[1];}
  // The engine may hold a faulted CUDA context: refuse further calls, and let process exit release it.
  void close() override {closed_=true;}
 private:
  int pipe_[2]={-1,-1};
  std::unique_ptr<WorkerCore> core_;
  bool closed_=false;
};

int whole(const std::string& option,const std::string& value) {
  if(value.empty() || value.size()>9) throw std::runtime_error(option+" must be an integer");
  for(char c:value) if(c<'0' || c>'9') throw std::runtime_error(option+" must be an integer");
  return std::stoi(value);
}
double real(const std::string& option,const std::string& value) {
  size_t used=0;double v=0;
  try {v=std::stod(value,&used);} catch(...) {throw std::runtime_error(option+" must be a number");}
  if(used!=value.size()) throw std::runtime_error(option+" must be a number");
  return v;
}

}  // namespace

int main(int argc,char** argv) try {
  if(argc<3 || argc%2!=1 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3-server --experimental MODEL_DIRECTORY [--token-file FILE] [--port N] ...");
  const std::string directory=argv[2];
  std::string token_path;int port=8080;
  lfm::http::Settings settings;
  std::set<std::string> seen;
  for(int i=3;i<argc;i+=2) {
    const std::string option=argv[i],value=argv[i+1];
    if(!seen.insert(option).second) throw std::runtime_error("duplicate option "+option);
    if(option=="--token-file") token_path=value;
    else if(option=="--port") port=whole(option,value);
    else if(option=="--request-timeout") settings.request_timeout=real(option,value);
    else if(option=="--socket-timeout") settings.socket_timeout=real(option,value);
    else if(option=="--session-idle-timeout") settings.session_idle_timeout=real(option,value);
    else if(option=="--engine-timeout") settings.engine_timeout=real(option,value);
    else if(option=="--total-capacity") settings.total_capacity=whole(option,value);
    else if(option=="--max-sessions") settings.max_sessions=whole(option,value);
    else if(option=="--queue-size") settings.queue_size=whole(option,value);
    else if(option=="--handlers") settings.handlers=whole(option,value);
    else throw std::runtime_error("unsupported option "+option);
  }
  std::string token;
  if(!token_path.empty()) token=lfm::http::token_file(token_path);
  else {
    const char* env=std::getenv("LFM3_SERVER_TOKEN");
    if(!env || !lfm::http::valid_token(env)) throw std::runtime_error("bearer token must be 32..256 URL-safe characters");
    token=env;
  }
  ::unsetenv("LFM3_SERVER_TOKEN");
  settings.validate();
  if(port<0 || port>65535) throw std::runtime_error("port must be an integer in 0..65535");

  // Signals go to one waiting thread, never to handler or engine threads.
  sigset_t signals;
  sigemptyset(&signals);sigaddset(&signals,SIGTERM);sigaddset(&signals,SIGINT);
  if(pthread_sigmask(SIG_BLOCK,&signals,nullptr)!=0) throw std::runtime_error("cannot block signals");
  std::signal(SIGPIPE,SIG_IGN);

  CoreBackend backend(directory);
  lfm::http::Policy policy;
  policy.vocabulary=backend.core().vocabulary();
  policy.suppress=backend.core().suppressed_tokens();
  policy.stops=backend.core().stop_tokens();
  for(auto* ids:{&policy.suppress,&policy.stops}) {
    std::sort(ids->begin(),ids->end());ids->erase(std::unique(ids->begin(),ids->end()),ids->end());
  }
  for(int id:policy.stops)
    if(std::binary_search(policy.suppress.begin(),policy.suppress.end(),id)) throw std::runtime_error("conflicting text generation policy");
  const auto& tokenizer=backend.core().tokenizer();  // loaded and checked before the first request

  auto service=std::make_unique<lfm::http::Service>(backend,tokenizer,policy,settings);
  auto server=std::make_unique<lfm::http::Server>(*service,token,port);
  token.assign(token.size(),'\0');
  // First SIGTERM/SIGINT: orderly shutdown. A second one: exit now.
  std::thread waiter([&] {
    int received=0;
    sigwait(&signals,&received);
    server->stop();
    sigwait(&signals,&received);
    std::_Exit(1);
  });
  std::cout<<"{\"ready\":true,\"port\":"<<server->port()<<"}\n"<<std::flush;
  server->serve();
  // Admission is closed: fail queued work, cancel running generations at their next token boundary,
  // wait for the engine owner, then give handlers one socket timeout to deliver their replies.
  service->close();
  server->drain(lfm::http::seconds(settings.socket_timeout));
  const int status=service->cleanup_failed()?1:0;
  std::cout<<std::flush;
  // Handler threads may still be writing replies to slow peers; they own no engine state. Exit without
  // running destructors under them.
  std::_Exit(status);
} catch(const std::exception& e) {std::cerr<<"lfm3-server: "<<e.what()<<"\n";return 1;}
