#pragma once
// A value handed from one thread to another under a key, for as long as someone holds a lease on it
// (validation/image-queue-v1 part 2: the in-process server's handler thread decodes an image, the engine-owner thread
// finds the pixels instead of decoding them again). The pattern of lfm::audio_features::FeatureCache
// (validation/audio-frontend-v1), for any value type.
// The key is the full input (an image part's base64 text): std::map compares whole keys, so two different inputs never
// share an entry, whatever they have in common; the same input shares one, which is safe because the value is a
// deterministic function of it. An entry lives while a lease on it lives: a request holds its leases until it ends,
// however it ends, so nothing outlives the requests in flight.
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace lfm {

template <class T>
class LeasedMap {
 public:
  using Value = std::shared_ptr<const T>;
  class Lease {
   public:
    Lease() = default;
    Lease(Lease&& o) noexcept : map_(o.map_), key_(std::move(o.key_)) { o.map_ = nullptr; }
    Lease& operator=(Lease&& o) noexcept {
      if (this != &o) release(), map_ = o.map_, key_ = std::move(o.key_), o.map_ = nullptr;
      return *this;
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { release(); }

   private:
    friend class LeasedMap;
    Lease(LeasedMap* map, std::string key) : map_(map), key_(std::move(key)) {}
    void release() noexcept {
      if (!map_) return;
      std::lock_guard g(map_->lock_);
      const auto it = map_->entries_.find(key_);
      if (it != map_->entries_.end() && --it->second.leases == 0) map_->entries_.erase(it);
      map_ = nullptr;
    }
    LeasedMap* map_ = nullptr;
    std::string key_;
  };
  // Leaves `value` under `key` (the one already there kept: the same input, the same value).
  Lease put(const std::string& key, Value value) {
    std::lock_guard g(lock_);
    auto& e = entries_[key];
    if (!e.value) e.value = std::move(value);
    ++e.leases;
    return Lease(this, key);
  }
  Value find(const std::string& key) const {
    std::lock_guard g(lock_);
    const auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second.value;
  }
  size_t size() const {
    std::lock_guard g(lock_);
    return entries_.size();
  }

 private:
  struct Entry {
    Value value;
    int leases = 0;
  };
  mutable std::mutex lock_;
  std::map<std::string, Entry> entries_;
};

}  // namespace lfm
