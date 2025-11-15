#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
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

  // Исследование, если создавать целую матрицу сразу в последовательной версии
  // std::vector<std::vector<InType>> matrix(n, std::vector<InType>(n));
  // for (InType i = 0; i < n; i++) {
  //  matrix[i][0] = 1;
  //  for (InType j = 1; j < n; j++) {
  //    matrix[i][j] = (i * n) + j + 1;
  //  }
  //}

  GetOutput().clear();
  GetOutput().reserve(n);

  for (InType i = 0; i < n; i++) {
    std::vector<InType> row(n);
    row[0] = 1;
    for (InType j = 1; j < n; j++) {
      row[j] = (i * n) + j + 1;
    }

    // Нахождение минимума в строке
    InType min_val = row[0];
    for (InType j = 1; j < n; j++) {
      min_val = std::min(min_val, row[j]);
    }
    GetOutput().push_back(min_val);
  }

  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

bool SabirovSMinValMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(GetInput()));
}

}  // namespace sabirov_s_min_val_matrix
