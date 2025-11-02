#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sabirov_s_min_val_matrix {

SabirovSMinValMatrixSEQ::SabirovSMinValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SabirovSMinValMatrixSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SabirovSMinValMatrixSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SabirovSMinValMatrixSEQ::RunImpl() {
  InType n = GetInput();
  if (n == 0) {
    return false;
  }

  std::vector<std::vector<InType>> matrix(n, std::vector<InType>(n));

  for (InType i = 0; i < n; i++) {
    matrix[i][0] = 1;
    for (InType j = 1; j < n; j++) {
      matrix[i][j] = i * n + j + 1;
    }
  }

  InType sum_of_mins = 0;
  for (InType i = 0; i < n; i++) {
    InType min_val = matrix[i][0];
    for (InType j = 1; j < n; j++) {
      if (matrix[i][j] < min_val) {
        min_val = matrix[i][j];
      }
    }
    sum_of_mins += min_val;
  }

  GetOutput() = sum_of_mins;
  return GetOutput() > 0;
}

bool SabirovSMinValMatrixSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace sabirov_s_min_val_matrix
