#include "sabirov_s_gauss_jordan_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"

namespace sabirov_s_gauss_jordan_method {

SabirovSGaussJordanMethodMPI::SabirovSGaussJordanMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSGaussJordanMethodMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Каждый процесс валидирует свои входные данные
  const auto &input = GetInput();

  // Проверяем, что входные данные не пустые
  if (input.empty()) {
    return false;
  }

  // Вычисляем размер системы (n уравнений, n неизвестных)
  size_t total_size = input.size();

  // Находим n: n * (n+1) = total_size
  double discriminant = 1.0 + 4.0 * static_cast<double>(total_size);
  if (discriminant < 0) {
    return false;
  }

  double n_double = (-1.0 + std::sqrt(discriminant)) / 2.0;
  size_t n = static_cast<size_t>(n_double);

  // Проверяем, что n - целое число и размер матрицы корректен
  if (std::abs(n_double - static_cast<double>(n)) > 1e-9 || n == 0) {
    return false;
  }

  if (n * (n + 1) != total_size) {
    return false;
  }

  return true;
}

bool SabirovSGaussJordanMethodMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &cached_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &cached_world_size_);

  const auto &input = GetInput();
  size_t total_size = input.size();
  double n_double = (-1.0 + std::sqrt(1.0 + 4.0 * static_cast<double>(total_size))) / 2.0;
  n_ = static_cast<size_t>(n_double);

  matrix_.assign(input.begin(), input.end());
  GetOutput().assign(n_, 0.0);

  const size_t cols = n_ + 1;
  pivot_row_buffer_.assign(cols, 0.0);

  row_displs_.assign(static_cast<size_t>(cached_world_size_) + 1, 0);
  row_counts_int_.assign(cached_world_size_, 0);
  row_displs_int_.assign(cached_world_size_, 0);

  size_t offset = 0;
  size_t base_rows = cached_world_size_ == 0 ? 0 : n_ / static_cast<size_t>(cached_world_size_);
  size_t remainder = cached_world_size_ == 0 ? 0 : n_ % static_cast<size_t>(cached_world_size_);
  for (int r = 0; r < cached_world_size_; ++r) {
    row_displs_[r] = offset;
    size_t rows = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
    row_counts_int_[r] = static_cast<int>(rows);
    row_displs_int_[r] = static_cast<int>(offset);
    offset += rows;
  }
  row_displs_[cached_world_size_] = offset;

  local_row_start_ = row_displs_[cached_rank_];
  local_row_end_ = row_displs_[cached_rank_ + 1];

  row_order_.resize(n_);
  row_position_.resize(n_);
  std::iota(row_order_.begin(), row_order_.end(), 0);
  for (size_t i = 0; i < n_; ++i) {
    row_position_[i] = i;
  }

  return true;
}

bool SabirovSGaussJordanMethodMPI::RunImpl() {
  const size_t cols = n_ + 1;

  if (cached_world_size_ == 1) {
    return RunSingleProcessGaussJordan();
  }

  constexpr double kEps = 1e-10;

  for (size_t step = 0; step < n_; ++step) {
    double local_max = 0.0;
    int local_row = -1;

    for (size_t pos = step; pos < n_; ++pos) {
      size_t row_id = row_order_[pos];
      if (row_id < local_row_start_ || row_id >= local_row_end_) {
        continue;
      }
      double value = std::abs(matrix_[row_id * cols + step]);
      if (value > local_max) {
        local_max = value;
        local_row = static_cast<int>(row_id);
      }
    }

    struct {
      double value;
      int row;
    } local_pair{local_max, local_row}, global_pair{};

    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (global_pair.row == -1 || global_pair.value < kEps) {
      return false;
    }

    size_t pivot_row_id = static_cast<size_t>(global_pair.row);
    size_t pivot_pos = row_position_[pivot_row_id];
    if (pivot_pos != step) {
      size_t other_row_id = row_order_[step];
      std::swap(row_order_[step], row_order_[pivot_pos]);
      row_position_[pivot_row_id] = step;
      row_position_[other_row_id] = pivot_pos;
    }

    pivot_row_id = row_order_[step];
    const int pivot_owner = GetRowOwner(pivot_row_id);

    if (cached_rank_ == pivot_owner) {
      double *row_ptr = &matrix_[pivot_row_id * cols];
      double pivot_value = row_ptr[step];
      for (size_t j = 0; j < cols; ++j) {
        pivot_row_buffer_[j] = row_ptr[j] / pivot_value;
      }
      std::copy(pivot_row_buffer_.begin(), pivot_row_buffer_.end(), row_ptr);
    }

    MPI_Bcast(pivot_row_buffer_.data(), static_cast<int>(cols), MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

    if (cached_rank_ != pivot_owner) {
      double *row_ptr = &matrix_[pivot_row_id * cols];
      std::copy(pivot_row_buffer_.begin(), pivot_row_buffer_.end(), row_ptr);
    }

    for (size_t pos = 0; pos < n_; ++pos) {
      size_t row_id = row_order_[pos];
      if (row_id == pivot_row_id) {
        continue;
      }
      if (row_id < local_row_start_ || row_id >= local_row_end_) {
        continue;
      }

      double *row_ptr = &matrix_[row_id * cols];
      double factor = row_ptr[step];
      if (std::abs(factor) < kEps) {
        row_ptr[step] = 0.0;
        continue;
      }
      for (size_t j = step; j < cols; ++j) {
        row_ptr[j] -= factor * pivot_row_buffer_[j];
      }
      row_ptr[step] = 0.0;
    }
  }

  size_t local_row_count = local_row_end_ - local_row_start_;
  std::vector<double> local_rhs(local_row_count, 0.0);
  for (size_t row = local_row_start_; row < local_row_end_; ++row) {
    local_rhs[row - local_row_start_] = matrix_[row * cols + n_];
  }

  std::vector<double> gathered_rhs(n_, 0.0);
  MPI_Allgatherv(local_rhs.data(), static_cast<int>(local_row_count), MPI_DOUBLE, gathered_rhs.data(),
                 row_counts_int_.data(), row_displs_int_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  for (size_t step = 0; step < n_; ++step) {
    size_t row_id = row_order_[step];
    GetOutput()[step] = gathered_rhs[row_id];
  }

  return true;
}

bool SabirovSGaussJordanMethodMPI::PostProcessingImpl() {
  // Постобработка не требуется
  return true;
}

bool SabirovSGaussJordanMethodMPI::RunSingleProcessGaussJordan() {
  const size_t cols = n_ + 1;
  constexpr double kEps = 1e-10;

  for (size_t i = 0; i < n_; ++i) {
    size_t pivot_row = i;
    double max_val = std::abs(matrix_[i * cols + i]);
    for (size_t k = i + 1; k < n_; ++k) {
      double val = std::abs(matrix_[k * cols + i]);
      if (val > max_val) {
        max_val = val;
        pivot_row = k;
      }
    }

    if (max_val < kEps) {
      return false;
    }

    if (pivot_row != i) {
      for (size_t j = 0; j < cols; ++j) {
        std::swap(matrix_[i * cols + j], matrix_[pivot_row * cols + j]);
      }
    }

    double *row_ptr = &matrix_[i * cols];
    double pivot = row_ptr[i];
    for (size_t j = i; j < cols; ++j) {
      row_ptr[j] /= pivot;
    }

    for (size_t k = 0; k < n_; ++k) {
      if (k == i) {
        continue;
      }
      double *other_row = &matrix_[k * cols];
      double factor = other_row[i];
      if (std::abs(factor) < kEps) {
        continue;
      }
      for (size_t j = i; j < cols; ++j) {
        other_row[j] -= factor * row_ptr[j];
      }
      other_row[i] = 0.0;
    }
  }

  for (size_t i = 0; i < n_; ++i) {
    GetOutput()[i] = matrix_[i * cols + n_];
  }
  return true;
}

int SabirovSGaussJordanMethodMPI::GetRowOwner(size_t global_row) const {
  auto it = std::upper_bound(row_displs_.begin(), row_displs_.end(), global_row);
  int idx = static_cast<int>(std::distance(row_displs_.begin(), it)) - 1;
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= cached_world_size_) {
    idx = cached_world_size_ - 1;
  }
  return idx;
}

}  // namespace sabirov_s_gauss_jordan_method
