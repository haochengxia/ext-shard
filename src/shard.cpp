#include "./shard.h"
#include <algorithm>
#include <random>
#include <xtensor/xbuilder.hpp>

ShardedStructure::ShardedStructure(const NestedList &nl) {
  num_ele_ = 0;
  max_shard_size_ = 0;

  nl_ = nl;
  num_shard_ = nl.size();
  for (auto s : nl) {
    int s_size = s.size();
    max_shard_size_ = std::max(s_size, max_shard_size_);
    num_ele_ += s_size;
  }

  // create mapping
  to_shard_idx_.resize(num_ele_);
  to_shard_size_.resize(num_ele_);

  int shard_idx = 0;
  for (auto &s : nl) {
    int s_size = s.size();
    for (int i : s) {
      to_shard_idx_[i] = shard_idx;
      to_shard_size_[i] = s_size;
    }
    shard_idx++;
  }

  // init idxes available
  idxes_available_ = xt::arange(num_ele_);
}

ShardedStructure::~ShardedStructure() {}

ShardedStructure::NestedList ShardedStructure::sample_perm_nest(unsigned seed) {
  std::mt19937 rng(seed);
  NestedList res(num_shard_);

  // outer shuffle
  std::vector<size_t> indices(num_shard_);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  for (size_t i = 0; i < num_shard_; ++i)
    res[i] = nl_[indices[i]];

  // inner shuffle
  for (auto &s : res) {
    if (s.size() > 1) {
      std::shuffle(s.begin(), s.end(), rng);
    }
  }
  return res;
}

ShardedStructure::IndexList ShardedStructure::sample_perm_flat(unsigned seed) {
  std::mt19937 rng(seed);
  IndexList res(num_ele_);

  std::vector<size_t> indices(num_shard_);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);

  int cursor = 0;
  for (size_t i = 0; i < num_shard_; i++) {
    auto s = nl_[indices[i]];
    std::shuffle(s.begin(), s.end(), rng);
    for (int v : s) {
      res[cursor] = v;
      cursor++;
    }
  }
  return res;
}
