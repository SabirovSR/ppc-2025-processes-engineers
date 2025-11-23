#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_gauss_jordan_method {

class SabirovSGaussJordanMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SabirovSGaussJordanMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные функции для уменьшения когнитивной сложности
  bool RunSingleProcessGaussJordan();
  [[nodiscard]] int GetRowOwner(size_t global_row) const;
  [[nodiscard]] std::pair<double, int> FindLocalPivot(size_t step, size_t cols) const;
  void UpdateRowOrder(size_t pivot_row_id, size_t step);
  void NormalizePivotRow(size_t pivot_row_id, size_t step, size_t cols);
  void BroadcastPivotRow(size_t pivot_row_id, int pivot_owner, size_t cols);
  void EliminateColumnMPI(size_t pivot_row_id, size_t step, size_t cols);

  // Вспомогательные функции для RunSingleProcessGaussJordan
  [[nodiscard]] size_t FindPivotRowSingle(size_t col, size_t cols) const;
  void SwapRowsSingle(size_t row1, size_t row2, size_t cols);
  void NormalizeRowSingle(size_t row, size_t col, size_t cols);
  void EliminateColumnSingle(size_t pivot_row, size_t col, size_t cols);

  std::vector<double> matrix_;
  size_t n_{0};
  std::vector<size_t> row_displs_;
  std::vector<int> row_counts_int_;
  std::vector<int> row_displs_int_;
  std::vector<size_t> row_order_;
  std::vector<size_t> row_position_;
  std::vector<double> pivot_row_buffer_;
  size_t local_row_start_{0};
  size_t local_row_end_{0};
  int cached_rank_{0};
  int cached_world_size_{1};
};

}  // namespace sabirov_s_gauss_jordan_method
