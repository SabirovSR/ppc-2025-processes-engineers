#pragma once

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

  bool RunSingleProcessGaussJordan();
  int GetRowOwner(size_t global_row) const;

  std::vector<double> matrix_;  // Рабочая матрица для метода Гаусса-Жордана
  size_t n_{0};                 // Размер системы (количество уравнений)
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
