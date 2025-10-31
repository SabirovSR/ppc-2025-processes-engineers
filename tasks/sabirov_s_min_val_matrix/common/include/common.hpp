#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sabirov_s_min_val_matrix {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sabirov_s_min_val_matrix
