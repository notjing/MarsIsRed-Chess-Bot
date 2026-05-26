#pragma once
#include "chess.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> boardParams(chess::Board board);
py::array_t<float> denseParams(chess::Board board);
