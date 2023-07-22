#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <random>
#include <vector>

namespace py = pybind11;

using NestedList = std::vector<std::vector<int>>;
using IndexList = std::vector<int>;
using std::nullopt;
using std::optional;

class ShardedStructure {
public:
  // ctor
  ShardedStructure(const NestedList &nl);
  // dtor
  ~ShardedStructure();

  // functions
  void set_random_state(unsigned seed);
  NestedList
  sample_perm_nest(optional<unsigned> seed = nullopt); // std::random_device()()
  IndexList sample_perm_flat(optional<unsigned> seed = nullopt);

  int get_max_shard_size() { return max_shard_size_; }
  int get_num_ele() { return num_ele_; }
  int get_num_shard() { return num_shard_; }
  int get_shard_idx(int ele_idx) { return to_shard_idx_[ele_idx]; }
  int get_shard_size(int ele_idx) { return to_shard_size_[ele_idx]; }

  // mutable
  IndexList idxes_available;

private:
  int max_shard_size_;
  int num_shard_;
  int num_ele_;
  NestedList nl_;
  unsigned seed_;

  std::mt19937 rng_;

  std::vector<int> to_shard_idx_;  // from ele index to its shard index
  std::vector<int> to_shard_size_; // from ele index to its shard size
};

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
  idxes_available.resize(num_ele_);
  std::iota(idxes_available.begin(), idxes_available.end(), 0);

  // init random state
  seed_ = rand();
  rng_.seed(seed_);
}

ShardedStructure::~ShardedStructure() {}

void ShardedStructure::set_random_state(unsigned seed) { rng_.seed(seed); }

NestedList ShardedStructure::sample_perm_nest(optional<unsigned> seed) {
  if (seed.has_value())
    rng_.seed(seed.value());

  NestedList res(num_shard_);
  // outer shuffle
  std::vector<size_t> indices(num_shard_);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng_);
  for (int i = 0; i < num_shard_; ++i)
    res[i] = nl_[indices[i]];

  // inner shuffle
  for (auto &s : res) {
    if (s.size() > 1) {
      std::shuffle(s.begin(), s.end(), rng_);
    }
  }
  return res;
}

IndexList ShardedStructure::sample_perm_flat(optional<unsigned> seed) {
  if (seed.has_value())
    rng_.seed(seed.value());

  IndexList res = IndexList(num_ele_);
  std::vector<size_t> indices(num_shard_);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng_);

  int cursor = 0;
  for (int i = 0; i < num_shard_; i++) {
    auto s = nl_[indices[i]];
    std::shuffle(s.begin(), s.end(), rng_);
    for (int v : s) {
      res[cursor] = v;
      cursor++;
    }
  }
  return res;
}

PYBIND11_MODULE(ext_shard, m) {
  // Define ShardedStructure
  py::class_<ShardedStructure>(m, "ShardedStructure")
      .def(py::init<NestedList &>())
      .def("set_random_state", &ShardedStructure::set_random_state,
           py::arg("seed") = std::random_device()())
      .def("sample_perm_nest", &ShardedStructure::sample_perm_nest,
           py::arg("seed") = py::none())
      .def("sample_perm_flat", &ShardedStructure::sample_perm_flat,
           py::arg("seed") = py::none())
      .def("get_max_shard_size", &ShardedStructure::get_max_shard_size)
      .def("get_num_ele", &ShardedStructure::get_num_ele)
      .def("get_num_shard", &ShardedStructure::get_num_shard)
      .def("get_shard_idx", &ShardedStructure::get_shard_idx)
      .def("get_shard_size", &ShardedStructure::get_shard_size)
      .def_readwrite("idxes_available", &ShardedStructure::idxes_available);
}
