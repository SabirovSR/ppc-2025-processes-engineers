#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sabirov_s_gauss_jordan_method {

// Input: расширенная матрица системы [A|b] в виде одномерного вектора
// Размер матрицы n x (n+1), где n - количество уравнений
using InType = std::vector<double>;
// Output: вектор решений системы уравнений
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sabirov_s_gauss_jordan_method
