// Independent-process wiring oracle: replay materialized logits, not model code.
#include <lfm/model_config.hpp>
#include <lfm/text_sampling.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

lfm::Json json(const std::string& path) {
  std::ifstream file(path);
  if(!file) throw std::runtime_error("cannot open "+path);
  return lfm::Json::parse(std::string(std::istreambuf_iterator<char>(file),{}));
}
std::vector<int> ids(const lfm::Json& root,const std::string& name) {
  std::vector<int> result;
  for(const auto& x:root.at(name).array()) result.push_back(x.integer());
  return result;
}
int main(int argc,char** argv) try {
  if(argc!=5) throw std::runtime_error("usage: text-sampling-oracle REQUEST.json RESULT.json TRACE_DIR VOCAB");
  const auto req=json(argv[1]),out=json(argv[2]);
  const int vocab=std::stoi(argv[4]);
  if(vocab<1 || vocab>16777216) throw std::runtime_error("invalid vocabulary");
  lfm::TextSamplingOptions options;
  options.do_sample=req.at("do_sample").boolean();
  options.temperature=req.at("temperature").number();
  options.top_k=req.at("top_k").integer();
  options.top_p=req.at("top_p").number();
  options.min_p=req.at("min_p").number();
  options.repetition_penalty=req.at("repetition_penalty").number();
  options.seed=req.at("seed").integer();
  if(!options.do_sample) throw std::runtime_error("oracle fixture must be stochastic");
  lfm::TextSampler sampler(options);
  const auto suppressed=ids(req,"suppress_tokens"),generated=ids(out,"generated_ids");
  if(generated.empty()) throw std::runtime_error("missing generation");
  std::vector<int> history;
  bool differs=false;
  for(size_t step=0;step<generated.size();++step) {
    std::ifstream file(std::string(argv[3])+"/"+std::to_string(step)+".logits.bf16",std::ios::binary);
    std::vector<std::uint16_t> raw(vocab);
    if(!file.read(reinterpret_cast<char*>(raw.data()),vocab*2) || file.peek()!=std::char_traits<char>::eof())
      throw std::runtime_error("missing/wrong-size BF16 trace");
    std::vector<float> logits(vocab);
    for(int i=0;i<vocab;++i) {std::uint32_t bits=std::uint32_t(raw[i])<<16; std::memcpy(&logits[i],&bits,sizeof(float));}
    int expected=sampler.sample(logits.data(),logits.size(),history,suppressed);
    if(expected!=generated[step]) throw std::runtime_error("sampler wiring mismatch at step "+std::to_string(step));
    auto greedy_options=options;greedy_options.do_sample=false;
    lfm::TextSampler greedy(greedy_options);
    differs |= expected!=greedy.sample(logits.data(),logits.size(),history,suppressed);
    history.push_back(expected);
  }
  if(!differs) throw std::runtime_error("fixture did not demonstrate non-greedy sampling");
  std::cout<<"PASS: replayed "<<generated.size()<<" sampled tokens; non-greedy choice verified\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n"; return 1;}
