// Oracle for lfm/audio_modes.hpp (A4): replays forced tokens and frames through the mode machine and prints the
// per-step trace (what was sampled, what was fed back, whether a decode runs, the suppression set), the emitted
// slots and frames, and the unconsumed inputs. stdin:
//   MODE(none|text|audio|interleaved) SUPPRESS_MIXED SUPPRESS_AUDIO_OUT RETURN_AUDIO_EOS MAX_NEW LAST_PROMPT_TOKEN
//   N_TOKENS tokens...   N_FRAMES then 12 codes per frame
#include <lfm/audio_modes.hpp>

#include <deque>
#include <iostream>
#include <string>

using namespace lfm::audio;

template <class C>
static void list(const C& c) {
  bool first = true;
  for (auto v : c) std::cout << (first ? "" : ",") << v, first = false;
}

int main() try {
  std::string mode;
  int sm, sa, re, max_new, last;
  std::cin >> mode >> sm >> sa >> re >> max_new >> last;
  ModeOptions o;
  o.mode = mode == "text" ? Requested::Text : mode == "audio" ? Requested::Audio
         : mode == "interleaved" ? Requested::Interleaved : Requested::None;
  o.suppress_mixed = sm, o.suppress_audio_out = sa, o.return_audio_eos = re, o.max_new_tokens = max_new;
  int n;
  std::cin >> n;
  std::deque<int> tokens{}; tokens.resize(size_t(n));
  for (int& t : tokens) std::cin >> t;
  std::cin >> n;
  std::deque<Frame> frames{}; frames.resize(size_t(n));
  for (Frame& f : frames) for (int& c : f) std::cin >> c;
  if (!std::cin) throw std::runtime_error("bad input");

  ModeMachine m(ModeTokens{}, o);
  std::cout << "marker " << m.begin(last) << "\n";
  std::vector<int> slots;
  std::vector<Frame> emitted;
  for (Step s; (s = m.next()) != Step::Done;) {
    Outcome r;
    if (s == Step::Text) {
      if (tokens.empty()) throw std::runtime_error("forced tokens exhausted");
      const int token = tokens.front();
      tokens.pop_front();
      const auto suppressed = m.suppressed();
      r = m.on_text(token);
      std::cout << "T " << token << " " << r.decode << " " << r.token << " ";
      list(suppressed);
      std::cout << "\n";
    } else {
      if (frames.empty()) throw std::runtime_error("forced frames exhausted");
      const Frame f = frames.front();
      frames.pop_front();
      r = m.on_frame(f);
      std::cout << "A ";
      list(f);
      std::cout << " " << r.decode << " ";
      list(r.fed);
      std::cout << "\n";
      if (r.emitted) emitted.push_back(r.fed);
    }
    if (r.emitted) slots.push_back(r.slot);
  }
  std::cout << "slots ";
  list(slots);
  std::cout << "\nframes " << emitted.size() << "\n";
  for (const Frame& f : emitted) list(f), std::cout << "\n";
  std::cout << "left " << tokens.size() << " " << frames.size() << "\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "audio_modes_oracle: " << e.what() << "\n";
  return 1;
}
