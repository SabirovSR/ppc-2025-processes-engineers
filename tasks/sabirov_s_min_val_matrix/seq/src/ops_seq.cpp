#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"

namespace sabirov_s_min_val_matrix {

SabirovSMinValMatrixSEQ::SabirovSMinValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SabirovSMinValMatrixSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput().empty());
}

bool SabirovSMinValMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool SabirovSMinValMatrixSEQ::RunImpl() {
  InType n = GetInput();
  if (n == 0) {
    return false;
  }

  GetOutput().clear();
  GetOutput().reserve(n);

  auto generate_value = [](int64_t i, int64_t j) -> InType {
    constexpr int64_t A = 1103515245;
    constexpr int64_t C = 12345;
    constexpr int64_t M = 2147483648;
    int64_t seed = (i * 100000007LL + j * 1000000009LL) ^ 42;
    int64_t val = (A * seed + C) % M;
    return static_cast<InType>((val % 2000001) - 1000000);
  };

  for (InType i = 0; i < n; i++) {
    InType min_val = generate_value(i, 0);
    for (InType j = 1; j < n; j++) {
      InType val = generate_value(i, j);
      min_val = std::min(min_val, val);
    }
    GetOutput().push_back(min_val);
  }

  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

bool SabirovSMinValMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(GetInput()));
}

}  // namespace sabirov_s_min_val_matrix
