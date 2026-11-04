// Single-threaded resident protocol prototype. Not network serving or GPU batching.
// Reuse the identical runtime in one translation unit; no second kernel fork.
#define LFM3_RUNTIME_ONLY
#include "lfm3_main.cu"
#include <lfm/text_sessions.hpp>
#include <map>
#include <iomanip>
#include <sstream>

namespace {
std::string quote(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\') out<<'\\'<<char(c);
    else if(c>=32 && c<127) out<<char(c);
    else {const char* hex="0123456789abcdef";out<<"\\u00"<<hex[c>>4]<<hex[c&15];}
  }
  out<<'"';return out.str();
}
std::string session_id(const lfm::Json& req) {
  auto id=req.at("session_id").string();
  if(id.empty() || id.size()>64) throw std::runtime_error("invalid session ID");
  for(char c:id) if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'))
    throw std::runtime_error("invalid session ID");
  return id;
}
struct Session {
  int capacity;
  std::unique_ptr<Runner> runner;
  std::vector<int> consumed;
  std::vector<float> last_scores;
};
lfm::TextSamplingOptions sampling_options(const lfm::Json& request,int vocab) {
  lfm::TextSamplingOptions o;
  if(request.has("do_sample")) o.do_sample=request.at("do_sample").boolean();
  if(request.has("temperature")) o.temperature=request.at("temperature").number();
  if(request.has("top_k")) o.top_k=request.at("top_k").integer();
  if(request.has("top_p")) o.top_p=request.at("top_p").number();
  if(request.has("min_p")) o.min_p=request.at("min_p").number();
  if(request.has("repetition_penalty")) o.repetition_penalty=request.at("repetition_penalty").number();
  if(request.has("seed")) o.seed=request.at("seed").integer();
  if(o.top_k>vocab) throw std::runtime_error("top_k exceeds vocabulary size");
  lfm::TextSampler validate(o);return o;
}
void schema(const lfm::Json& request,const std::string& op) {
  const std::set<std::string> common={"op","session_id"};
  const std::set<std::string> generate={"input_ids","max_new_tokens","do_sample","temperature","top_k","top_p","min_p","repetition_penalty","seed","return_logits"};
  const auto* object=std::get_if<lfm::Json::Object>(&request.value);
  if(!object) throw std::runtime_error("expected object");
  for(const auto& [key,value]:*object) {
    if(common.count(key)) continue;
    if(op=="create" && key=="capacity") continue;
    if(op=="generate" && generate.count(key)) continue;
    throw std::runtime_error("unsupported request field: "+key);
  }
}
} // namespace

int main(int argc,char** argv) try {
  if(argc!=3 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3-worker --experimental MODEL_DIRECTORY");
  const std::string directory=argv[2];
  const auto model=read_json(directory+"/config.json");
  auto cfg=lfm::ModelConfig::load(directory+"/config.json",32768);
  if(!cfg.is_lfm3() || !cfg.sandwich_norm || !cfg.tied || cfg.shape.hidden!=2048 || cfg.shape.head_dim!=128 || cfg.shape.conv_width!=7)
    throw std::runtime_error("unsupported worker geometry");
  std::vector<int> suppressed,stops=cfg.eos;
  if(cfg.architecture==lfm::ModelConfig::Architecture::Lfm3Omni) {
    for(auto name:{"audio_token_id","audio_out_token_id","image_token_id","video_token_id","mixed_start_token_id","audio_start_token_id"})
      suppressed.push_back(model.at(name).integer());
    stops.push_back(model.at("text_end_token_id").integer());
  }
  for(int id:suppressed) if(id>=cfg.shape.vocabulary) throw std::runtime_error("invalid modality token");
  for(int id:stops) if(id>=cfg.shape.vocabulary) throw std::runtime_error("invalid stop token");
  std::shared_ptr<lfm::Safetensors> weights;
  std::map<std::string,Session> sessions;
  lfm::text::SessionLedger ledger(2,32768);
  // JSON readers generally parse numbers as double. Emit enough digits to
  // preserve the exact float->double value, not merely a float32 roundtrip.
  std::cout<<std::setprecision(std::numeric_limits<double>::max_digits10);
  std::cout<<"{\"ready\":true,\"protocol\":1,\"max_sessions\":2,\"aggregate_capacity\":32768,\"gpu_batching\":false}\n"<<std::flush;
  for(;;) {
    // Bounded framing, unlike unbounded std::getline allocation.
    std::string line;bool oversized=false;char c;
    while(std::cin.get(c) && c!='\n') {
      if(line.size()<4*1024*1024) line+=c;else oversized=true;
    }
    if(line.empty() && !std::cin && !oversized) break;
    try {
      if(oversized) throw std::runtime_error("request exceeds 4 MiB");
      const auto request=lfm::Json::parse(line);
      const auto op=request.at("op").string();schema(request,op);
      if(op=="shutdown") {
        std::cout<<"{\"ok\":true,\"shutdown\":true}\n"<<std::flush;break;
      }
      if(op=="stats") {
        std::cout<<"{\"ok\":true,\"sessions\":"<<ledger.size()<<",\"reserved_tokens\":"<<ledger.reserved_tokens()
                 <<",\"weights_loaded\":"<<(weights?"true":"false")<<"}\n"<<std::flush;continue;
      }
      const auto id=session_id(request);
      if(op=="create") {
        int capacity=request.at("capacity").integer();lfm::text::scratch_rows(capacity);
        ledger.create(id,capacity);
        try {sessions.emplace(id,Session{capacity,{},{},{}});}
        catch(...) {ledger.close(id);throw;}
        std::cout<<"{\"ok\":true,\"session_id\":"<<quote(id)<<"}\n"<<std::flush;continue;
      }
      auto found=sessions.find(id);
      if(found==sessions.end()) throw std::runtime_error("unknown session");
      auto& session=found->second;
      if(op=="inspect") {
        const auto snapshot=ledger.inspect(id);
        std::cout<<"{\"ok\":true,\"capacity\":"<<snapshot.capacity<<",\"consumed_tokens\":"<<snapshot.consumed.size()
                 <<",\"transcript_tokens\":"<<snapshot.transcript.size()<<",\"cache_valid\":"<<(snapshot.cache_valid?"true":"false")
                 <<",\"inflight\":"<<(snapshot.inflight?"true":"false")<<"}\n"<<std::flush;continue;
      }
      if(op=="close") {
        ledger.close(id);sessions.erase(found);
        std::cout<<"{\"ok\":true,\"closed\":true}\n"<<std::flush;continue;
      }
      if(op=="reset") {
        try {if(session.runner) session.runner->reset();}
        catch(...) {session.runner.reset();session.consumed.clear();session.last_scores.clear();ledger.reset(id);throw;}
        session.consumed.clear();session.last_scores.clear();
        ledger.reset(id);
        std::cout<<"{\"ok\":true,\"reset\":true}\n"<<std::flush;continue;
      }
      if(op!="generate") throw std::runtime_error("unknown operation");
      const auto input=ids(request,"input_ids",cfg.shape.vocabulary);
      for(int token:input) if(std::find(suppressed.begin(),suppressed.end(),token)!=suppressed.end())
        throw std::runtime_error("modality placeholder input is unsupported");
      const int count=request.at("max_new_tokens").integer();
      if(count<1) throw std::runtime_error("max_new_tokens must be positive");
      lfm::text::require_budget(input.size(),count,session.capacity);
      const bool return_logits=request.has("return_logits") && request.at("return_logits").boolean();
      const auto options=sampling_options(request,cfg.shape.vocabulary);
      lfm::TextSampler sampler(options);
      auto lease=ledger.begin(id,lfm::text::SessionLedger::Tokens(input.begin(),input.end()),count);
      // Every metadata rejection above is pre-mutation. Execution errors below
      // invalidate the entire sequence cache; no partially written cache survives.
      try {
        if(!weights) weights=std::make_shared<lfm::Safetensors>(directory+"/model.safetensors");
        if(!session.runner) {
          auto shape=cfg;shape.shape.context=session.capacity;
          session.runner=std::make_unique<Runner>(directory,shape,weights);
        }
        auto& runner=*session.runner;
        const bool reuse=!lease.plan().full_prefill;
        const size_t reused=lease.plan().reuse_tokens;
        if(reuse && (session.consumed.size()!=reused || runner.cursor!=int(reused)))
          throw std::runtime_error("session physical cache/ledger disagreement");
        if(!reuse) {runner.reset();session.consumed.clear();session.last_scores.clear();}
        if(!reuse) {
          session.last_scores=runner.forward(input);session.consumed=input;
        } else for(size_t i=reused;i<input.size();++i) {
          session.last_scores=runner.forward({input[i]});session.consumed.push_back(input[i]);
        }
        std::vector<int> generated;
        for(int t=0;t<count;++t) {
          int token=sampler.sample(session.last_scores.data(),session.last_scores.size(),generated,suppressed);
          generated.push_back(token);
          if(std::find(stops.begin(),stops.end(),token)!=stops.end() || t==count-1) break;
          session.last_scores=runner.forward({token});session.consumed.push_back(token);
        }
        lease.commit(lfm::text::SessionLedger::Tokens(generated.begin(),generated.end()),
                     lfm::text::SessionLedger::Tokens(session.consumed.begin(),session.consumed.end()));
        std::cout<<"{\"ok\":true,\"session_id\":"<<quote(id)<<",\"generated_ids\":";print_ids(std::cout,generated);
        std::cout<<",\"reused_tokens\":"<<reused<<",\"consumed_tokens\":"<<session.consumed.size()
                 <<",\"do_sample\":"<<(options.do_sample?"true":"false")<<",\"seed\":"<<options.seed;
        if(return_logits) {
          std::cout<<",\"last_logits\":[";
          for(size_t i=0;i<session.last_scores.size();++i) {if(i)std::cout<<',';std::cout<<session.last_scores[i];}
          std::cout<<']';
        }
        std::cout<<"}\n"<<std::flush;
      } catch(...) {
        session.runner.reset();session.consumed.clear();session.last_scores.clear();throw;
      }
    } catch(const std::exception& e) {
      std::cout<<"{\"ok\":false,\"error\":"<<quote(e.what())<<"}\n"<<std::flush;
    }
  }
  return 0;
} catch(const std::exception& e) {std::cerr<<"lfm3-worker: "<<e.what()<<"\n";return 1;}
