// The fields liquid-lfm3-worker (WorkerCore) accepts, per op. A field outside this list fails the request
// ("unsupported request field: KEY"), which the Service treats as a worker failure (503 for everything after).
// Shared with the host harness's fake engine (native/tests/lfm3_http_harness.cpp), which runs every frame the
// Service sends through it, so a field the Service starts sending is refused on the host exactly as on Thor
// (audio-handles-v1 window 19: "audio_ref" was missing here, and only the fake engine knew it).
#pragma once
#include <lfm/model_config.hpp>

#include <set>
#include <stdexcept>
#include <string>
#include <variant>

namespace lfm::worker {

inline void schema(const lfm::Json& request, const std::string& op) {
  static const std::set<std::string> common = {"op", "session_id", "rpc_id"};
  static const std::set<std::string> generate = {"input_ids", "max_new_tokens", "do_sample", "temperature", "top_k",
                                                 "top_p", "min_p", "repetition_penalty", "seed", "return_logits",
                                                 "trace_sampling", "messages", "return_text"};
  static const std::set<std::string> speech = {"chunks", "mode", "max_new_tokens", "seed", "chunked",
                                               "audio", "audio_ref", "images", "stream"};
  const auto* object = std::get_if<lfm::Json::Object>(&request.value);
  if (!object) throw std::runtime_error("expected object");
  for (const auto& [key, value] : *object) {
    if (common.count(key)) continue;
    if (op == "create" && key == "capacity") continue;
    if ((op == "generate" || op == "text_begin") && generate.count(key)) continue;
    if (op == "text_step" && (key == "tokens" || key == "cancel")) continue;
    if (op == "generate_batch" && key == "lanes") continue;
    if ((op == "speech" || op == "speech_admit") && speech.count(key)) continue;
    if (op == "speech_admit" && key == "ticket") continue;
    if (op == "speech_step" && key == "cancel") continue;
    throw std::runtime_error("unsupported request field: " + key);
  }
}

}  // namespace lfm::worker
