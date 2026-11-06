// Single-threaded resident protocol prototype; explicit two-lane decode groups.
// Reuse the identical runtime in one translation unit; no second kernel fork.
#define LFM3_RUNTIME_ONLY
#include "lfm3_main.cu"
#include "lfm3_decode_batch.cuh"
#include <lfm/text_sessions.hpp>
#include <lfm/text_cancellation.hpp>
#include <lfm/text_batch_cancellation.hpp>
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
struct GenerationCancelled : std::exception {
  const char* what() const noexcept override {return "generation cancelled at a token boundary";}
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
  const std::set<std::string> common={"op","session_id","rpc_id"};
  const std::set<std::string> generate={"input_ids","max_new_tokens","do_sample","temperature","top_k","top_p","min_p","repetition_penalty","seed","return_logits","trace_sampling"};
  const auto* object=std::get_if<lfm::Json::Object>(&request.value);
  if(!object) throw std::runtime_error("expected object");
  for(const auto& [key,value]:*object) {
    if(common.count(key)) continue;
    if(op=="create" && key=="capacity") continue;
    if(op=="generate" && generate.count(key)) continue;
    if(op=="generate_batch" && (key=="op" || key=="rpc_id" || key=="lanes")) continue;
    throw std::runtime_error("unsupported request field: "+key);
  }
}
// Metadata is collected for BOTH lanes before any lease, weight load or reset.
struct BatchLane {
  std::string id;
  Session* session=nullptr;
  std::vector<int> input,generated;
  int count=0,paired=0;
  bool return_logits=false,trace=false,done=false,cancelled=false;
  size_t reused=0;
  lfm::TextSamplingOptions options;
  std::unique_ptr<lfm::TextSampler> sampler;
  std::unique_ptr<lfm::text::SessionLedger::Lease> lease;
};
void generate_batch(const lfm::Json& request,int rpc_id,const std::string& directory,
                    const lfm::ModelConfig& cfg,const std::vector<int>& suppressed,
                    const std::vector<int>& stops,const std::string& trace_root,
                    std::shared_ptr<lfm::Safetensors>& weights,
                    std::map<std::string,Session>& sessions,lfm::text::SessionLedger& ledger,
                    lfm::text::CancellationReader& cancellation) {
  // Outer schema is intentionally narrower than the single-operation common set.
  for(const auto& [key,value]:std::get<lfm::Json::Object>(request.value))
    if(key!="op" && key!="rpc_id" && key!="lanes") throw std::runtime_error("unsupported batch field: "+key);
  const auto& requests=request.at("lanes").array();
  if(requests.size()!=2) throw std::runtime_error("generate_batch requires exactly two lanes");
  std::array<BatchLane,2> lanes;
  for(size_t i=0;i<2;++i) {
    const auto& req=requests[i];auto& lane=lanes[i];
    schema(req,"generate");
    if(req.has("op") || req.has("rpc_id")) throw std::runtime_error("lane cannot carry op or rpc_id");
    lane.id=session_id(req);
    auto found=sessions.find(lane.id);
    if(found==sessions.end()) throw std::runtime_error("unknown session");
    lane.session=&found->second;
    lane.input=ids(req,"input_ids",cfg.shape.vocabulary);
    for(int token:lane.input) if(std::find(suppressed.begin(),suppressed.end(),token)!=suppressed.end())
      throw std::runtime_error("modality placeholder input is unsupported");
    lane.count=req.at("max_new_tokens").integer();
    if(lane.count<1) throw std::runtime_error("max_new_tokens must be positive");
    lfm::text::require_budget(lane.input.size(),lane.count,lane.session->capacity);
    lane.return_logits=req.has("return_logits") && req.at("return_logits").boolean();
    lane.trace=req.has("trace_sampling") && req.at("trace_sampling").boolean();
    if(lane.trace && (trace_root.empty() || lane.count>32))
      throw std::runtime_error("sampling trace needs startup audit root and at most32 output tokens");
    lane.options=sampling_options(req,cfg.shape.vocabulary);
    lane.sampler=std::make_unique<lfm::TextSampler>(lane.options);
    if(ledger.inspect(lane.id).inflight) throw std::runtime_error("session already inflight");
  }
  if(lanes[0].id==lanes[1].id) throw std::runtime_error("batch sessions must be distinct");
  const auto trace_key=std::to_string(rpc_id);
  const auto trace_directory=trace_root+"/"+trace_key;
  auto discard=[](BatchLane& lane) {
    lane.session->runner.reset();lane.session->consumed.clear();lane.session->last_scores.clear();
    if(lane.lease) lane.lease->abort();
  };
  auto poll=[&] {
    const auto cancelled=cancellation.requested_pair(rpc_id);
    for(size_t i=0;i<2;++i) if(!lanes[i].done && cancelled[i]) {
      discard(lanes[i]);lanes[i].cancelled=true;lanes[i].done=true;
    }
  };
  int paired=0;
  try {
    for(auto& lane:lanes) lane.lease=std::make_unique<lfm::text::SessionLedger::Lease>(
      ledger.begin(lane.id,lfm::text::SessionLedger::Tokens(lane.input.begin(),lane.input.end()),lane.count));
    poll();
    if((lanes[0].trace || lanes[1].trace) && !std::filesystem::create_directory(trace_directory))
      throw std::runtime_error("sampling trace request already exists");
    // Prefills and reuse continuations remain serial, never decode-batched.
    for(size_t i=0;i<2;++i) {
      auto& lane=lanes[i];if(lane.done) continue;
      auto& session=*lane.session;
      if(!weights) weights=std::make_shared<lfm::Safetensors>(directory+"/model.safetensors");
      if(!session.runner) {
        auto shape=cfg;shape.shape.context=session.capacity;
        session.runner=std::make_unique<Runner>(directory,shape,weights);
      }
      const bool reuse=!lane.lease->plan().full_prefill;
      lane.reused=lane.lease->plan().reuse_tokens;
      if(reuse && (session.consumed.size()!=lane.reused || session.runner->cursor!=int(lane.reused)))
        throw std::runtime_error("session physical cache/ledger disagreement");
      if(!reuse) {session.runner->reset();session.consumed.clear();session.last_scores.clear();}
      poll(); // initialization/reset may have been long: check BEFORE prefill.
      if(lane.done) continue;
      if(!reuse) {
        session.last_scores=session.runner->forward(lane.input);session.consumed=lane.input;
      } else for(size_t j=lane.reused;j<lane.input.size();++j) {
        poll();if(lane.done) break;
        session.last_scores=session.runner->forward({lane.input[j]});session.consumed.push_back(lane.input[j]);
      }
      poll(); // also latch cancellation for the peer while this lane prefills.
      if(lane.trace && !std::filesystem::create_directory(trace_directory+"/lane"+std::to_string(i)))
        throw std::runtime_error("sampling trace lane already exists");
    }
    while(!lanes[0].done || !lanes[1].done) {
      poll();
      std::vector<size_t> ready;
      std::vector<Runner*> owners;std::vector<int> tokens;
      for(size_t i=0;i<2;++i) {
        auto& lane=lanes[i];if(lane.done) continue;
        auto& session=*lane.session;
        if(lane.trace) {
          std::vector<bf16> raw(session.last_scores.size());
          for(size_t j=0;j<raw.size();++j) raw[j]=__float2bfloat16_rn(session.last_scores[j]);
          std::ofstream file(trace_directory+"/lane"+std::to_string(i)+"/"+std::to_string(lane.generated.size())+".logits.bf16",std::ios::binary);
          file.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(bf16));
          if(!file) throw std::runtime_error("sampling trace write failed");
        }
        const int token=lane.sampler->sample(session.last_scores.data(),session.last_scores.size(),lane.generated,suppressed);
        lane.generated.push_back(token);
        if(std::find(stops.begin(),stops.end(),token)!=stops.end() || lane.generated.size()==size_t(lane.count)) {
          poll();if(lane.done) continue;
          lane.lease->commit(lfm::text::SessionLedger::Tokens(lane.generated.begin(),lane.generated.end()),
                             lfm::text::SessionLedger::Tokens(session.consumed.begin(),session.consumed.end()));
          lane.done=true; // completed lanes are never touched by poll/discard again.
        } else ready.push_back(i);
      }
      poll();
      for(auto i:ready) if(!lanes[i].done) {
        owners.push_back(lanes[i].session->runner.get());tokens.push_back(lanes[i].generated.back());
      }
      if(owners.empty()) continue;
      std::vector<std::vector<float>> scores;
      if(owners.size()==2) {scores=lfm3_decode_batch::forward(owners,tokens);++paired;}
      else scores.push_back(owners[0]->forward({tokens[0]}));
      size_t row=0;
      for(auto i:ready) if(!lanes[i].done) {
        auto& lane=lanes[i];lane.session->last_scores=std::move(scores[row++]);
        lane.session->consumed.push_back(lane.generated.back());
        if(owners.size()==2) ++lane.paired;
      }
    }
  } catch(...) {
    // No CUDA fault isolation promise: fail closed for ALL still-inflight lanes.
    for(auto& lane:lanes) if(!lane.done && lane.lease) discard(lane);
    throw;
  }
  std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"gpu_batching\":"<<(paired?"true":"false")
           <<",\"paired_decode_calls\":"<<paired<<",\"results\":[";
  for(size_t i=0;i<2;++i) {
    const auto& lane=lanes[i];const auto& session=*lane.session;
    if(i) std::cout<<',';
    std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(lane.id)<<",\"generated_ids\":";
    print_ids(std::cout,lane.generated);
    std::cout<<",\"reused_tokens\":"<<(lane.cancelled?0:lane.reused)<<",\"consumed_tokens\":"<<session.consumed.size()
             <<",\"do_sample\":"<<(lane.options.do_sample?"true":"false")<<",\"seed\":"<<lane.options.seed
             <<",\"cancelled\":"<<(lane.cancelled?"true":"false")<<",\"cache_valid\":"<<(lane.cancelled?"false":"true")
             <<",\"gpu_batching\":"<<(lane.paired?"true":"false")<<",\"paired_decode_calls\":"<<lane.paired;
    if(lane.trace) std::cout<<",\"sampling_trace\":"<<quote(trace_key+"/lane"+std::to_string(i));
    if(lane.return_logits && !lane.cancelled) {
      std::cout<<",\"last_logits\":[";
      for(size_t j=0;j<session.last_scores.size();++j) {if(j) std::cout<<',';std::cout<<session.last_scores[j];}
      std::cout<<']';
    }
    std::cout<<'}';
  }
  std::cout<<"]}\n"<<std::flush;
}
} // namespace

int main(int argc,char** argv) try {
  if(argc<3 || argc%2!=1 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3-worker --experimental MODEL_DIRECTORY [--sampling-trace-root NEW_DIRECTORY] [--cancel-fd FD]");
  const std::string directory=argv[2];
  std::string sampling_trace_root;int cancellation_fd=-1;
  std::set<std::string> options_seen;
  for(int i=3;i<argc;i+=2) {
    std::string option=argv[i];
    if(!options_seen.insert(option).second) throw std::runtime_error("duplicate worker option");
    if(option=="--sampling-trace-root") sampling_trace_root=argv[i+1];
    else if(option=="--cancel-fd") {
      std::string value=argv[i+1];size_t used=0;
      cancellation_fd=std::stoi(value,&used);
      if(used!=value.size() || cancellation_fd<3) throw std::runtime_error("invalid cancellation fd");
    } else throw std::runtime_error("unsupported worker option");
  }
  lfm::text::CancellationReader cancellation(cancellation_fd);
  if(options_seen.count("--sampling-trace-root") && (sampling_trace_root.empty() || !std::filesystem::create_directory(sampling_trace_root)))
    throw std::runtime_error("sampling trace root must be a new directory");
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
   std::cout<<"{\"ready\":true,\"protocol\":2,\"max_sessions\":2,\"aggregate_capacity\":32768,\"gpu_batching\":false,\"decode_batching\":true,\"cancellation\":"
           <<(cancellation_fd>=0?"true":"false")<<"}\n"<<std::flush;
  int last_rpc_id=0;
  for(;;) {
    // Bounded framing, unlike unbounded std::getline allocation.
    std::string line;bool oversized=false;char c;
    while(std::cin.get(c) && c!='\n') {
      if(line.size()<4*1024*1024) line+=c;else oversized=true;
    }
    if(line.empty() && !std::cin && !oversized) break;
    int rpc_id=0;
    try {
      if(oversized) throw std::runtime_error("request exceeds 4 MiB");
      const auto request=lfm::Json::parse(line);
      rpc_id=request.at("rpc_id").integer();
      if(rpc_id<1 || last_rpc_id==2147483647 || rpc_id!=last_rpc_id+1)
        throw std::runtime_error("out-of-sequence RPC ID");
      last_rpc_id=rpc_id;
      const auto op=request.at("op").string();schema(request,op);
      if(op=="shutdown") {
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"shutdown\":true}\n"<<std::flush;break;
      }
       if(op=="stats") {
         size_t free_bytes=0,total_bytes=0;
         if(weights) {cuda_check(cudaDeviceSynchronize());cuda_check(cudaMemGetInfo(&free_bytes,&total_bytes));}
         std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"sessions\":"<<ledger.size()<<",\"reserved_tokens\":"<<ledger.reserved_tokens()
                  <<",\"weights_loaded\":"<<(weights?"true":"false");
         // Device-wide unified-memory accounting, NOT per-worker allocations or RSS.
         if(weights) std::cout<<",\"cuda_free_bytes\":"<<free_bytes<<",\"cuda_total_bytes\":"<<total_bytes
                             <<",\"device_allocated_bytes\":"<<(total_bytes-free_bytes);
         std::cout<<"}\n"<<std::flush;continue;
      }
       if(op=="generate_batch") {
         generate_batch(request,rpc_id,directory,cfg,suppressed,stops,sampling_trace_root,weights,sessions,ledger,cancellation);
         continue;
       }
       const auto id=session_id(request);
      if(op=="create") {
        int capacity=request.at("capacity").integer();lfm::text::scratch_rows(capacity);
        ledger.create(id,capacity);
        try {sessions.emplace(id,Session{capacity,{},{},{}});}
        catch(...) {ledger.close(id);throw;}
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)<<"}\n"<<std::flush;continue;
      }
      auto found=sessions.find(id);
      if(found==sessions.end()) throw std::runtime_error("unknown session");
      auto& session=found->second;
      if(op=="inspect") {
        const auto snapshot=ledger.inspect(id);
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"capacity\":"<<snapshot.capacity<<",\"consumed_tokens\":"<<session.consumed.size()
                 <<",\"committed_consumed_tokens\":"<<snapshot.consumed.size()
                 <<",\"transcript_tokens\":"<<snapshot.transcript.size()<<",\"cache_valid\":"<<(snapshot.cache_valid?"true":"false")
                 <<",\"inflight\":"<<(snapshot.inflight?"true":"false")<<"}\n"<<std::flush;continue;
      }
      if(op=="close") {
        ledger.close(id);sessions.erase(found);
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"closed\":true}\n"<<std::flush;continue;
      }
      if(op=="reset") {
        try {if(session.runner) session.runner->reset();}
        catch(...) {session.runner.reset();session.consumed.clear();session.last_scores.clear();ledger.reset(id);throw;}
        session.consumed.clear();session.last_scores.clear();
        ledger.reset(id);
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"reset\":true}\n"<<std::flush;continue;
      }
      if(op!="generate") throw std::runtime_error("unknown operation");
      const auto input=ids(request,"input_ids",cfg.shape.vocabulary);
      for(int token:input) if(std::find(suppressed.begin(),suppressed.end(),token)!=suppressed.end())
        throw std::runtime_error("modality placeholder input is unsupported");
      const int count=request.at("max_new_tokens").integer();
      if(count<1) throw std::runtime_error("max_new_tokens must be positive");
      lfm::text::require_budget(input.size(),count,session.capacity);
      const bool return_logits=request.has("return_logits") && request.at("return_logits").boolean();
      const bool trace_sampling=request.has("trace_sampling") && request.at("trace_sampling").boolean();
      if(trace_sampling && (sampling_trace_root.empty() || count>32))
        throw std::runtime_error("sampling trace needs startup audit root and at most32 output tokens");
      const auto options=sampling_options(request,cfg.shape.vocabulary);
      lfm::TextSampler sampler(options);
      auto lease=ledger.begin(id,lfm::text::SessionLedger::Tokens(input.begin(),input.end()),count);
      std::vector<int> generated;
      auto check_cancellation=[&] {
        if(cancellation.requested(rpc_id)) throw GenerationCancelled{};
      };
      // Every metadata rejection above is pre-mutation. Execution errors below
      // invalidate the entire sequence cache; no partially written cache survives.
      try {
        check_cancellation();
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
          session.last_scores=lfm::text::at_cancellation_boundary(check_cancellation,[&] {return runner.forward(input);});
          session.consumed=input;
        } else for(size_t i=reused;i<input.size();++i) {
          check_cancellation();
          session.last_scores=runner.forward({input[i]});session.consumed.push_back(input[i]);
        }
        const auto sampling_trace=sampling_trace_root+"/"+std::to_string(rpc_id);
        if(trace_sampling && !std::filesystem::create_directory(sampling_trace))
          throw std::runtime_error("sampling trace request already exists");
        for(int t=0;t<count;++t) {
          check_cancellation();
          if(trace_sampling) {
            std::vector<bf16> trace_scores(session.last_scores.size());
            for(size_t i=0;i<trace_scores.size();++i) trace_scores[i]=__float2bfloat16_rn(session.last_scores[i]);
            std::ofstream trace_file(sampling_trace+"/"+std::to_string(t)+".logits.bf16",std::ios::binary);
            trace_file.write(reinterpret_cast<const char*>(trace_scores.data()),trace_scores.size()*sizeof(bf16));
            if(!trace_file) throw std::runtime_error("sampling trace write failed");
          }
          int token=sampler.sample(session.last_scores.data(),session.last_scores.size(),generated,suppressed);
          generated.push_back(token);
          if(std::find(stops.begin(),stops.end(),token)!=stops.end() || t==count-1) break;
          session.last_scores=runner.forward({token});session.consumed.push_back(token);
        }
        check_cancellation();
        lease.commit(lfm::text::SessionLedger::Tokens(generated.begin(),generated.end()),
                     lfm::text::SessionLedger::Tokens(session.consumed.begin(),session.consumed.end()));
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)<<",\"generated_ids\":";print_ids(std::cout,generated);
        std::cout<<",\"reused_tokens\":"<<reused<<",\"consumed_tokens\":"<<session.consumed.size()
                  <<",\"do_sample\":"<<(options.do_sample?"true":"false")<<",\"seed\":"<<options.seed;
        if(trace_sampling) std::cout<<",\"sampling_trace\":"<<rpc_id;
        if(return_logits) {
          std::cout<<",\"last_logits\":[";
          for(size_t i=0;i<session.last_scores.size();++i) {if(i)std::cout<<',';std::cout<<session.last_scores[i];}
          std::cout<<']';
        }
        std::cout<<"}\n"<<std::flush;
      } catch(const GenerationCancelled&) {
        session.runner.reset();session.consumed.clear();session.last_scores.clear();lease.abort();
        std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)
                 <<",\"cancelled\":true,\"cache_valid\":false,\"consumed_tokens\":0,\"reused_tokens\":0,\"generated_ids\":";
        print_ids(std::cout,generated);
        std::cout<<",\"do_sample\":"<<(options.do_sample?"true":"false")<<",\"seed\":"<<options.seed<<"}\n"<<std::flush;
      } catch(...) {
        session.runner.reset();session.consumed.clear();session.last_scores.clear();throw;
      }
    } catch(const std::exception& e) {
      std::cout<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":false,\"error\":"<<quote(e.what())<<"}\n"<<std::flush;
    }
  }
  return 0;
} catch(const std::exception& e) {std::cerr<<"lfm3-worker: "<<e.what()<<"\n";return 1;}
