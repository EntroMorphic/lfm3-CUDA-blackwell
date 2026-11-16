// validation/image-cache-v1 rule 3 (host): lfm::LruBytes and lfm::LeasedMap under their contracts. Exits nonzero with a
// message on the first violation. Built by tests/test_lru_bytes.py (ASan+UBSan; TSan for the threaded part).
#include <lfm/leased_map.hpp>
#include <lfm/lru_bytes.hpp>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::fprintf(stderr, "FAILED %s:%d %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

int main() {
  // 200 distinct keys of random sizes under a budget: the total never exceeds it; the most recent entry is kept
  {
    std::mt19937 rng(1117);
    const size_t budget = 64 << 10;
    lfm::LruBytes<int> lru(budget);
    std::vector<std::string> keys;
    for (int i = 0; i < 200; ++i) {
      std::string k(1 + rng() % 4000, 'a' + i % 26);
      k += std::to_string(i);
      const size_t value_bytes = rng() % 8000;
      lru.put(k, std::make_shared<const int>(i), value_bytes);
      CHECK(lru.bytes() <= budget);
      if (k.size() + value_bytes <= budget) CHECK(lru.find(k) && *lru.find(k) == i);
      keys.push_back(k);
    }
    CHECK(lru.size() > 1);
  }
  // least recently used goes first; a find makes an entry recent
  {
    lfm::LruBytes<int> lru(30);
    lru.put("a", std::make_shared<const int>(1), 9);  // 10 each
    lru.put("b", std::make_shared<const int>(2), 9);
    lru.put("c", std::make_shared<const int>(3), 9);
    CHECK(lru.find("a"));                              // a recent: b is the oldest
    lru.put("d", std::make_shared<const int>(4), 9);
    CHECK(!lru.find("b") && lru.find("a") && lru.find("c") && lru.find("d") && lru.bytes() == 30);
    // the same key again keeps the first value
    lru.put("a", std::make_shared<const int>(99), 9);
    CHECK(*lru.find("a") == 1 && lru.bytes() == 30);
    // an entry larger than the budget is not kept, and evicts nothing
    lru.put("huge", std::make_shared<const int>(5), 100);
    CHECK(!lru.find("huge") && lru.size() == 3);
    // keys differing in their last byte are different entries
    lru.set_budget(1000);
    lru.put(std::string(100, 'x') + "1", std::make_shared<const int>(7), 0);
    lru.put(std::string(100, 'x') + "2", std::make_shared<const int>(8), 0);
    CHECK(*lru.find(std::string(100, 'x') + "1") == 7 && *lru.find(std::string(100, 'x') + "2") == 8);
    // an evicted value lives on for whoever holds it
    auto held = lru.find("c");
    lru.set_budget(0);
    CHECK(lru.size() == 0 && lru.bytes() == 0 && held && *held == 3);
  }
  // budget 0 keeps nothing
  {
    lfm::LruBytes<int> lru(0);
    lru.put("a", std::make_shared<const int>(1), 0);
    CHECK(lru.size() == 0);
  }
  // LeasedMap: an entry lives while a lease lives; the same key shares one entry; threads put and find at once
  {
    lfm::LeasedMap<int> m;
    {
      auto l1 = m.put("k", std::make_shared<const int>(1));
      {
        auto l2 = m.put("k", std::make_shared<const int>(2));
        CHECK(*m.find("k") == 1 && m.size() == 1);
      }
      CHECK(m.find("k"));
      auto moved = std::move(l1);
      CHECK(m.find("k"));
    }
    CHECK(!m.find("k") && m.size() == 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&m, t] {
        for (int i = 0; i < 2000; ++i) {
          const std::string key = "key" + std::to_string((t + i) % 5);
          auto lease = m.put(key, std::make_shared<const int>(i));
          CHECK(m.find(key));
        }
      });
    for (auto& t : threads) t.join();
    CHECK(m.size() == 0);
  }
  // LockedLruBytes from several threads: the budget holds
  {
    lfm::LockedLruBytes<int> lru;
    lru.set_budget(10000);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&lru, t] {
        for (int i = 0; i < 2000; ++i) {
          lru.put("k" + std::to_string(t * 100000 + i), std::make_shared<const int>(i), size_t(i % 300));
          lru.find("k" + std::to_string(t * 100000 + i / 2));
          CHECK(lru.bytes() <= 10000);
        }
      });
    for (auto& t : threads) t.join();
  }
  std::puts("lru_bytes_check: PASS");
}
