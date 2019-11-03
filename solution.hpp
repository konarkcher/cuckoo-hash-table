#pragma once

#include <tpcc/concurrency/backoff.hpp>

#include <tpcc/stdlike/atomic.hpp>
#include <tpcc/stdlike/mutex.hpp>

#include <tpcc/logging/logging.hpp>

#include <tpcc/support/compiler.hpp>
#include <tpcc/support/random.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace tpcc {
namespace solutions {

////////////////////////////////////////////////////////////////////////////////

class TTASSpinLock {
 public:
  TTASSpinLock() = default;

  TTASSpinLock(const TTASSpinLock&) = delete;
  TTASSpinLock(TTASSpinLock&&) = delete;

  void lock() {
    Backoff backoff{};

    do {
      while (locked_.load()) {
        backoff();
      }
    } while (locked_.exchange(true));
  }

  void unlock() {
    locked_.store(false);
  }

 private:
  tpcc::atomic<bool> locked_{false};
};

////////////////////////////////////////////////////////////////////////////////

using Mutex = TTASSpinLock;
using MutexLocker = std::unique_lock<Mutex>;

class LockSet {
 public:
  LockSet& Lock(Mutex& mutex) {
    locks_.emplace_back(mutex);
    return *this;
  }

  void Unlock() {
    locks_.clear();
  }

 private:
  std::vector<MutexLocker> locks_;
};

class LockManager {
 public:
  LockManager(size_t concurrency_level) : locks_(concurrency_level) {
  }

  size_t GetLockCount() const {
    return locks_.size();
  }

  LockSet Lock(size_t i) {
    return std::move(LockSet{}.Lock(locks_[i]));
  }

  LockSet Lock(size_t i, size_t j) {
    if (i > j) {
      std::swap(i, j);
    }
    if (i != j) {
      return std::move(Lock(i).Lock(locks_[j]));
    }
    return Lock(i);
  }

  LockSet LockAllFrom(size_t start) {
    LockSet lock_set{};
    for (size_t i = start; i < locks_.size(); ++i) {
      lock_set.Lock(locks_[i]);
    }
    return lock_set;
  }

 private:
  std::vector<Mutex> locks_;
};

////////////////////////////////////////////////////////////////////////////////

struct ElementHash {
  size_t primary_hash_;
  size_t alt_hash_;

  bool operator==(const ElementHash& that) const {
    return primary_hash_ == that.primary_hash_;
  }

  void Alternate() {
    std::swap(primary_hash_, alt_hash_);
  }
};

template <typename T, typename HashFunction>
class CuckooHasher {
 public:
  ElementHash operator()(const T& element) const {
    size_t hash_value = hasher_(element);
    return {.primary_hash_ = hash_value,
            .alt_hash_ = ComputeAltValue(hash_value) ^ hash_value};
  }

 private:
  uint16_t ComputeAltValue(size_t hash_value) const {
    hash_value *= 1349110179037u;
    return ((hash_value >> 48) ^ (hash_value >> 32) ^ (hash_value >> 16) ^
            hash_value) &
           0xFFFF;
  }

 private:
  std::hash<T> hasher_;
};

////////////////////////////////////////////////////////////////////////////////

struct CuckooPathSlot {
  size_t bucket_;
  size_t slot_;

  CuckooPathSlot(size_t bucket, size_t slot) : bucket_(bucket), slot_(slot) {
  }
};

using CuckooPath = std::vector<CuckooPathSlot>;

////////////////////////////////////////////////////////////////////////////////

// use this exceptions to interrupt and retry current operation

struct TableOvercrowded : std::logic_error {
  TableOvercrowded() : std::logic_error("Table overcrowded") {
  }
};

struct TableExpanded : std::logic_error {
  TableExpanded() : std::logic_error("Table expanded") {
  }
};

////////////////////////////////////////////////////////////////////////////////

template <typename T, class HashFunction = std::hash<T>>
class CuckooHashSet {
 private:
  // tune thresholds
  const size_t kMaxPathLength = 10;
  const size_t kFindPathIterations = 16;
  const size_t kEvictIterations = 16;

 private:
  struct GhostSlot {
    const T& element_;
    ElementHash hash_;

    GhostSlot(const T& element, const ElementHash& hash)
        : element_(element), hash_(hash) {
    }
  };

  struct CuckooSlot {
    T element_;
    ElementHash hash_;
    bool occupied_{false};

    template <typename E>
    void Set(E&& element, const ElementHash& hash) {
      element_ = std::forward<E>(element);
      hash_ = hash;
      occupied_ = true;
    }

    void Set(CuckooSlot&& slot) {
      Set(std::move(slot.element_), slot.hash_);
      slot.Clear();
    }

    void Clear() {
      occupied_ = false;
    }

    bool IsOccupied() const {
      return occupied_;
    }

    bool operator==(const GhostSlot& elem) const {
      return IsOccupied() && hash_ == elem.hash_ && element_ == elem.element_;
    }
  };

  using CuckooBucket = std::vector<CuckooSlot>;
  using CuckooBuckets = std::vector<CuckooBucket>;

  using TwoBuckets = std::pair<size_t, size_t>;
  using SlotPair = std::pair<CuckooSlot&, CuckooSlot&>;

 public:
  explicit CuckooHashSet(const size_t concurrency_level = 32,
                         const size_t bucket_width = 4)
      : lock_manager_(concurrency_level),
        kBucketWidth_(bucket_width),
        buckets_(CreateEmptyTable(concurrency_level)) {
  }

  template <typename E>
  bool Insert(E&& element) {
    const auto slot = CreateGhostSlot(element);

    while (true) {
      try {
        auto bucket_locks = LockTwoBuckets(slot.hash_);
        const auto buckets = GetBuckets(slot.hash_);

        if (Contains(buckets, slot)) {
          return false;
        }
        if (TryInsert(buckets, std::forward<E>(element), slot.hash_)) {
          return true;
        }

        RememberBucketCount();
        bucket_locks.Unlock();

        EvictElement(buckets);
      } catch (const TableExpanded& _) {
        // LOG_SIMPLE("Insert interrupted due to concurrent table expansion");
      } catch (const TableOvercrowded& _) {
        // LOG_SIMPLE("Cannot insert element, table overcrowded");
        ExpandTable();
      }
    }

    UNREACHABLE();
  }

  bool Remove(const T& element) {
    auto slot = CreateGhostSlot(element);
    const auto bucket_locks = LockTwoBuckets(slot.hash_);
    const auto buckets = GetBuckets(slot.hash_);

    for (auto bucket : {buckets.first, buckets.second}) {
      const auto pos = Find(buckets_[bucket], slot);
      if (pos != kNotFound) {
        buckets_[bucket][pos].Clear();
        size_.fetch_sub(1);
        return true;
      }

      slot.hash_.Alternate();
    }
    return false;
  }

  bool Contains(const T& element) const {
    const auto slot = CreateGhostSlot(element);
    const auto bucket_locks = LockTwoBuckets(slot.hash_);
    const auto buckets = GetBuckets(slot.hash_);

    return Contains(buckets, slot);
  }

  size_t GetSize() const {
    return size_.load();
  }

  double GetLoadFactor() const {
    const auto lock = lock_manager_.Lock(0);
    return size_.load() / (buckets_.size() * kBucketWidth_);
  }

 private:
  CuckooBuckets CreateEmptyTable(const size_t bucket_count) {
    return CuckooBuckets(bucket_count, CuckooBucket(kBucketWidth_));
  }

  size_t HashToBucket(const size_t hash_value) const {
    return hash_value % buckets_.size();
  }

  TwoBuckets GetBuckets(const ElementHash& hash) const {
    return {HashToBucket(hash.primary_hash_), HashToBucket(hash.alt_hash_)};
  }

  void RememberBucketCount() const {
    // store current bucket count in thread local storage
    expected_bucket_count_ = buckets_.size();
  }

  void InterruptIfTableExpanded() const {
    if (buckets_.size() != expected_bucket_count_) {
      throw TableExpanded{};
    }
  }

  size_t GetLockIndex(const size_t hash_value) const {
    return hash_value % lock_manager_.GetLockCount();
  }

  LockSet LockBucket(const size_t bucket) const {
    return lock_manager_.Lock(GetLockIndex(bucket));
  }

  LockSet LockTwoBuckets(const ElementHash& hash) const {
    return lock_manager_.Lock(GetLockIndex(hash.primary_hash_),
                              GetLockIndex(hash.alt_hash_));
  }

  GhostSlot CreateGhostSlot(const T& element) const {
    return {element, hasher_(element)};
  }

  CuckooSlot& GetSlot(const CuckooPathSlot& slot) {
    return buckets_[slot.bucket_][slot.slot_];
  }

  void MoveElement(CuckooSlot& from, CuckooSlot& to) {
    to.Set(std::move(from));
    to.hash_.Alternate();
    from.Clear();
  }

  // returns false if conflict detected during optimistic move
  bool TryMoveHoleBackward(const CuckooPath& path) {
    for (auto it = path.rbegin(); it + 1 != path.rend(); ++it) {
      const auto locks = LockTwoBuckets({it->bucket_, (it + 1)->bucket_});
      InterruptIfTableExpanded();

      SlotPair slots{GetSlot(*it), GetSlot(*(it + 1))};
      if (slots.first.IsOccupied()) {
        return false;
      }
      if (!slots.second.IsOccupied()) {
        continue;
      }
      if (HashToBucket(slots.second.hash_.alt_hash_) != it->bucket_) {
        return false;
      }

      MoveElement(slots.second, slots.first);
    }
    return true;
  }

  size_t GetRandomSlot() const {
    return tpcc::RandomUInteger(kBucketWidth_ - 1);
  }

  bool TryFindCuckooPathWithRandomWalk(const size_t start_bucket,
                                       CuckooPath& path) const {
    auto current_bucket = start_bucket;
    for (size_t i = 0; i < kMaxPathLength; ++i) {
      const auto bucket_lock = LockBucket(current_bucket);
      InterruptIfTableExpanded();

      const auto& bucket = buckets_[current_bucket];
      const auto empty_slot = FindEmptySlot(bucket);

      if (empty_slot != kNotFound) {
        path.emplace_back(current_bucket, empty_slot);
        return true;
      }
      path.emplace_back(current_bucket, GetRandomSlot());
      current_bucket = HashToBucket(bucket[path.back().slot_].hash_.alt_hash_);
    }
    return false;
  }

  size_t GetRandomBucket(const TwoBuckets& buckets) {
    return tpcc::TossFairCoin() ? buckets.first : buckets.second;
  }

  CuckooPath FindCuckooPath(const TwoBuckets& start_buckets) {
    for (size_t i = 0; i < kFindPathIterations; ++i) {
      const auto start_bucket = GetRandomBucket(start_buckets);
      CuckooPath path;
      if (TryFindCuckooPathWithRandomWalk(start_bucket, path)) {
        return path;
      }
    }

    throw TableOvercrowded{};
  }

  void EvictElement(const TwoBuckets& buckets) {
    for (size_t i = 0; i < kEvictIterations; ++i) {
      const auto path = FindCuckooPath(buckets);
      if (TryMoveHoleBackward(path)) {
        return;
      }
    }

    throw TableOvercrowded{};
  }

  void ExpandTable() {
    const auto first_lock = lock_manager_.Lock(0);
    if (buckets_.size() != expected_bucket_count_) {
      return;
    }
    const auto locks = lock_manager_.LockAllFrom(1);

    auto old_buckets = CreateEmptyTable(buckets_.size() * kGrowFactor_);
    buckets_.swap(old_buckets);

    std::vector<size_t> empty_slot(buckets_.size(), 0);

    for (auto& old_bucket : old_buckets) {
      for (auto& slot : old_bucket) {
        if (slot.IsOccupied()) {
          const auto bucket = HashToBucket(slot.hash_.primary_hash_);
          buckets_[bucket][empty_slot[bucket]++].Set(std::move(slot));
        }
      }
    }
  }

  size_t Find(const CuckooBucket& bucket, const GhostSlot& elem_slot) const {
    const auto iter = std::find(bucket.begin(), bucket.end(), elem_slot);
    return iter == bucket.end() ? kNotFound : iter - bucket.begin();
  }

  static bool IsEmpty(const CuckooSlot& slot) {
    return !slot.IsOccupied();
  }

  size_t FindEmptySlot(const CuckooBucket& bucket) const {
    const auto iter = std::find_if(bucket.begin(), bucket.end(), IsEmpty);
    return iter == bucket.end() ? kNotFound : iter - bucket.begin();
  }

  template <typename E>
  bool TryInsert(const TwoBuckets& buckets, E&& elem, const ElementHash& hash) {
    auto current_hash = hash;

    for (auto bucket : {buckets.first, buckets.second}) {
      const auto pos = FindEmptySlot(buckets_[bucket]);
      if (pos != kNotFound) {
        buckets_[bucket][pos].Set(std::forward<E>(elem), current_hash);
        size_.fetch_add(1);
        return true;
      }

      current_hash.Alternate();
    }
    return false;
  }

  bool Contains(const TwoBuckets& buckets, GhostSlot elem_slot) const {
    if (Find(buckets_[buckets.first], elem_slot) != kNotFound) {
      return true;
    }
    elem_slot.hash_.Alternate();
    return Find(buckets_[buckets.second], elem_slot) != kNotFound;
  }

 private:
  CuckooHasher<T, HashFunction> hasher_;

  mutable LockManager lock_manager_;

  const size_t kBucketWidth_;  // number of slots per bucket
  const size_t kGrowFactor_{2};
  const size_t kNotFound = std::numeric_limits<size_t>::max();

  tpcc::atomic<size_t> size_{0};
  CuckooBuckets buckets_;

  static thread_local size_t expected_bucket_count_;
};

template <typename T, typename HashFunction>
thread_local size_t CuckooHashSet<T, HashFunction>::expected_bucket_count_ = 0;

}  // namespace solutions
}  // namespace tpcc
