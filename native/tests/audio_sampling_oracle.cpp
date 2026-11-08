// Oracle for lfm/audio_sampling.hpp (B2). Reads stdin, writes stdout; one mode per call:
//   dist CODEBOOK       stdin: 2017 logits, H, then H history codes -> "code probability" lines (exact)
//   codes N SEED CODEBOOK   same stdin -> 2017 counts of N draws
//   order N SEED        -> 144 counts [position][codebook], then 66 counts of codebook pairs sharing a pass
//   frame N SEED        stdin: 2017 logits (every codebook scores them) -> checks the pass protocol; prints
//                       12 x 2017 counts of the frame's codes
//   window              accepts 65 frames, prints codebook 0's history (the window keeps the last 64)
#include <lfm/audio_sampling.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace lfm::audio;

static void read_case(std::vector<float>& logits, FrameSampler& sampler, int codebook) {
  logits.resize(kCodes);
  for (auto& v : logits) std::cin >> v;
  int h;
  std::cin >> h;
  for (int i = 0; i < h; ++i) {
    Frame f{};
    std::cin >> f[size_t(codebook)];
    sampler.accept(f);
  }
}

int main(int argc, char** argv) try {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "dist" && argc == 3) {
    FrameSampler sampler({0.4, 1.2, 64, 0});
    std::vector<float> logits;
    const int c = std::atoi(argv[2]);
    read_case(logits, sampler, c);
    std::cout << std::setprecision(18);
    for (const auto& d : sampler.distribution(c, logits.data())) std::cout << d.token_id << " " << d.probability << "\n";
  } else if (mode == "codes" && argc == 5) {
    FrameSampler sampler({0.4, 1.2, 64, std::strtoull(argv[3], nullptr, 10)});
    std::vector<float> logits;
    const int c = std::atoi(argv[4]);
    read_case(logits, sampler, c);
    std::vector<long> counts(kCodes);
    for (long n = std::atol(argv[2]); n--;) ++counts[size_t(sampler.sample_code(c, logits.data()))];
    for (long v : counts) std::cout << v << "\n";
  } else if (mode == "order" && argc == 4) {
    FrameSampler sampler({0.4, 1.2, 64, std::strtoull(argv[3], nullptr, 10)});
    std::vector<long> position(144), together(66);
    for (long n = std::atol(argv[2]); n--;) {
      const auto o = sampler.order();
      int pass[kCodebooks];
      for (int i = 0; i < kCodebooks; ++i) ++position[size_t(i * 12 + o[i])], pass[o[i]] = i / 4;
      for (int a = 0, k = 0; a < kCodebooks; ++a)
        for (int b = a + 1; b < kCodebooks; ++b, ++k) together[size_t(k)] += pass[a] == pass[b];
    }
    for (long v : position) std::cout << v << "\n";
    for (long v : together) std::cout << v << "\n";
  } else if (mode == "frame" && argc == 4) {
    FrameSampler sampler({0.4, 1.2, 64, std::strtoull(argv[3], nullptr, 10)});
    std::vector<float> fixed(kCodes);
    for (auto& v : fixed) std::cin >> v;
    std::vector<long> counts(size_t(kCodebooks) * kCodes);
    for (long n = std::atol(argv[2]); n--;) {
      int call = 0, revealed = 0;
      bool was_masked[kCodebooks];
      for (bool& m : was_masked) m = true;
      const Frame f = sampler.sample_frame([&](const int* codes, const bool* masked, const bool* picked, float* out) {
        int picks = 0, still = 0;
        for (int c = 0; c < kCodebooks; ++c) {
          if (picked[c] && !masked[c]) throw std::runtime_error("picked a revealed codebook");
          if (!masked[c] && was_masked[c]) throw std::runtime_error("revealed outside a pass");
          if (masked[c] != was_masked[c]) throw std::runtime_error("mask changed between passes unexpectedly");
          picks += picked[c];
          still += masked[c];
          (void)codes;
          if (picked[c]) std::copy(fixed.begin(), fixed.end(), out + size_t(c) * kCodes);
        }
        if (picks != kSchedule[size_t(call)] || still != kCodebooks - revealed)
          throw std::runtime_error("pass protocol");
        for (int c = 0; c < kCodebooks; ++c) if (picked[c]) was_masked[c] = false;
        revealed += picks;
        ++call;
      });
      if (call != 3) throw std::runtime_error("expected 3 passes");
      for (int c = 0; c < kCodebooks; ++c) ++counts[size_t(c) * kCodes + size_t(f[size_t(c)])];
    }
    for (long v : counts) std::cout << v << "\n";
  } else if (mode == "window" && argc == 2) {
    FrameSampler sampler;
    for (int i = 0; i < 65; ++i) {
      Frame f{};
      f[0] = i;
      sampler.accept(f);
    }
    for (int v : sampler.history_codes(0)) std::cout << v << "\n";
    try {
      Frame eos{};
      eos[3] = kEosCode;
      sampler.accept(eos);
      std::cout << "accepted-eos\n";
    } catch (const std::invalid_argument&) {
      std::cout << "refused-eos\n";
    }
  } else {
    std::cerr << "bad arguments\n";
    return 2;
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "audio_sampling_oracle: " << e.what() << "\n";
  return 1;
}
