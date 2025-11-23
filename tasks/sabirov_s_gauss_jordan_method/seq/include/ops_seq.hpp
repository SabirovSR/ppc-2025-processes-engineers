#pragma once

#include <cstddef>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_gauss_jordan_method {

class SabirovSGaussJordanMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SabirovSGaussJordanMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные функции для уменьшения когнитивной сложности
  [[nodiscard]] size_t FindPivotRow(size_t col, size_t cols) const;
  void SwapRows(size_t row1, size_t row2, size_t cols);
  void NormalizeRow(size_t row, size_t col, size_t cols);
  void EliminateColumn(size_t pivot_row, size_t col, size_t cols);

  std::vector<double> matrix_;  // Рабочая матрица для метода Гаусса-Жордана
  size_t n_{0};                 // Размер системы (количество уравнений)
};

}  // namespace sabirov_s_gauss_jordan_method
