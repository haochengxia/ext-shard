#include "pybind11/pybind11.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include "./shard.h"
#include <cmath>
#include <iostream>
#include <numeric>

namespace py = pybind11;

// utils

PYBIND11_MODULE(xtensor_shard, m) {
  xt::import_numpy();

  m.doc() = R"pbdoc(
        An xtensor extension for shard structure

        .. currentmodule:: xtensor_shard

        .. autosummary::
           :toctree: _generate

           example1
           example2
           readme_example1
           vectorize_example1
    )pbdoc";

  m.def("example1", example1,
        "Return the first element of an array, of dimension at least one");
  m.def("example2", example2, "Return the the specified array plus 2");

  m.def("readme_example1", readme_example1,
        "Accumulate the sines of all the values of the specified array");

  m.def("vectorize_example1", xt::pyvectorize(scalar_func),
        "Add the sine and and cosine of the two specified values");
}
