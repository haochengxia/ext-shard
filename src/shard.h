#include <iostream>
#include <random>
#include <vector>
#include <xtensor/xarray.hpp>

class ShardedStructure {
public:
  using NestedList = std::vector<std::vector<int>>;
  using IndexList = xt::xarray<int>;

  // ctor
  ShardedStructure(const NestedList &nl);
  // dtor
  ~ShardedStructure();

  // functions
  NestedList sample_perm_nest(unsigned seed = std::random_device()());
  IndexList sample_perm_flat(unsigned seed = std::random_device()());

  int get_max_shard_size() { return max_shard_size_; }
  int get_num_ele() { return num_ele_; }
  int get_shard_idx(int ele_idx) { return to_shard_idx_[ele_idx]; }
  int get_shard_size(int ele_idx) { return to_shard_size_[ele_idx]; }

  // mutable
  IndexList idxes_available_;

private:
  int max_shard_size_;
  int num_shard_;
  int num_ele_;
  NestedList nl_;

  std::vector<int> to_shard_idx_;  // from ele index to its shard index
  std::vector<int> to_shard_size_; // from ele index to its shard size
};
