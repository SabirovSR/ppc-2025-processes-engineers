#include "sabirov_s_gauss_jordan_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"

namespace sabirov_s_gauss_jordan_method {

namespace {
constexpr double kEps = 1e-10;
}  // namespace

SabirovSGaussJordanMethodMPI::SabirovSGaussJordanMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSGaussJordanMethodMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  // Валидация только на ранге 0, результат рассылается всем
  int valid = 0;
  if (rank_ == 0) {
    const auto &input = GetInput();

    if (!input.empty()) {
      size_t total_size = input.size();
      double discriminant = 1.0 + (4.0 * static_cast<double>(total_size));

      if (discriminant >= 0) {
        double n_double = (-1.0 + std::sqrt(discriminant)) / 2.0;
        auto n_val = static_cast<size_t>(n_double);

        if (std::abs(n_double - static_cast<double>(n_val)) <= 1e-9 && n_val > 0 && n_val * (n_val + 1) == total_size) {
          valid = 1;
        }
      }
    }
  }

  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid == 1;
}

bool SabirovSGaussJordanMethodMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  // Ранг 0 вычисляет размерности и рассылает их
  if (rank_ == 0) {
    const auto &input = GetInput();
    size_t total_size = input.size();
    double n_double = (-1.0 + std::sqrt(1.0 + (4.0 * static_cast<double>(total_size)))) / 2.0;
    n_ = static_cast<size_t>(n_double);
    cols_ = n_ + 1;

    // Сохраняем полную матрицу на ранге 0
    full_matrix_.assign(input.begin(), input.end());
  }

  // Рассылаем размерности всем процессам
  MPI_Bcast(&n_, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  cols_ = n_ + 1;

  // Инициализируем выходной вектор
  GetOutput().assign(n_, 0.0);

  // Вычисляем распределение строк по процессам
  row_counts_.resize(world_size_);
  row_displs_.resize(world_size_);
  send_counts_.resize(world_size_);
  send_displs_.resize(world_size_);

  size_t base_rows = n_ / static_cast<size_t>(world_size_);
  size_t remainder = n_ % static_cast<size_t>(world_size_);

  size_t offset = 0;
  for (int proc = 0; proc < world_size_; ++proc) {
    size_t rows = base_rows + (static_cast<size_t>(proc) < remainder ? 1 : 0);
    row_counts_[proc] = static_cast<int>(rows);
    row_displs_[proc] = static_cast<int>(offset);
    send_counts_[proc] = static_cast<int>(rows * cols_);
    send_displs_[proc] = static_cast<int>(offset * cols_);
    offset += rows;
  }

  local_row_count_ = static_cast<size_t>(row_counts_[rank_]);
  local_row_start_ = static_cast<size_t>(row_displs_[rank_]);

  // Выделяем память под локальные строки
  local_matrix_.resize(local_row_count_ * cols_);
  pivot_row_buf_.resize(cols_);

  // Инициализируем перестановку строк (изначально identity)
  row_permutation_.resize(n_);
  for (size_t i = 0; i < n_; ++i) {
    row_permutation_[i] = i;
  }

  // Распределяем данные от ранга 0
  DistributeMatrixFromRoot();

  return true;
}

void SabirovSGaussJordanMethodMPI::DistributeMatrixFromRoot() {
  // Ранг 0 отправляет каждому процессу его строки
  MPI_Scatterv(rank_ == 0 ? full_matrix_.data() : nullptr, send_counts_.data(), send_displs_.data(), MPI_DOUBLE,
               local_matrix_.data(), static_cast<int>(local_row_count_ * cols_), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // После распределения ранг 0 может освободить полную матрицу
  if (rank_ == 0) {
    full_matrix_.clear();
    full_matrix_.shrink_to_fit();
  }
}

void SabirovSGaussJordanMethodMPI::GatherResultsToRoot() {
  // Собираем правые части (последний столбец) со всех процессов
  std::vector<double> local_rhs(local_row_count_);
  for (size_t i = 0; i < local_row_count_; ++i) {
    local_rhs[i] = local_matrix_[(i * cols_) + n_];
  }

  std::vector<double> gathered_rhs;
  if (rank_ == 0) {
    gathered_rhs.resize(n_);
  }

  MPI_Gatherv(local_rhs.data(), static_cast<int>(local_row_count_), MPI_DOUBLE, gathered_rhs.data(), row_counts_.data(),
              row_displs_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Ранг 0 восстанавливает порядок решения согласно перестановкам
  if (rank_ == 0) {
    for (size_t step = 0; step < n_; ++step) {
      size_t global_row = row_permutation_[step];
      GetOutput()[step] = gathered_rhs[global_row];
    }
  }

  // Рассылаем результат всем процессам
  MPI_Bcast(GetOutput().data(), static_cast<int>(n_), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool SabirovSGaussJordanMethodMPI::RunImpl() {
  if (world_size_ == 1) {
    return RunSingleProcessGaussJordan();
  }

  return RunDistributedGaussJordan();
}

bool SabirovSGaussJordanMethodMPI::RunDistributedGaussJordan() {
  for (size_t step = 0; step < n_; ++step) {
    // 1. Поиск глобального максимального pivot элемента
    auto [local_max, local_row] = FindLocalPivot(step);

    struct {
      double value;
      int row;
    } local_pair{.value = local_max, .row = local_row}, global_pair{};

    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    // 2. Проверка на вырожденность
    if (global_pair.row == -1 || global_pair.value < kEps) {
      return false;
    }

    // 3. Обновляем перестановку строк
    auto global_pivot_row = static_cast<size_t>(global_pair.row);
    if (row_permutation_[step] != global_pivot_row) {
      // Находим позицию global_pivot_row в перестановке
      for (size_t pos = step + 1; pos < n_; ++pos) {
        if (row_permutation_[pos] == global_pivot_row) {
          std::swap(row_permutation_[step], row_permutation_[pos]);
          break;
        }
      }
    }

    // 4. Нормализация pivot строки и её рассылка
    int pivot_owner = GetRowOwner(global_pivot_row);
    NormalizeAndBroadcastPivotRow(global_pivot_row, pivot_owner, step);

    // 5. Элиминация столбца на всех локальных строках
    EliminateColumnLocal(step);
  }

  // Собираем результаты
  GatherResultsToRoot();

  return true;
}

std::pair<double, int> SabirovSGaussJordanMethodMPI::FindLocalPivot(size_t step) const {
  double local_max = 0.0;
  int local_row = -1;

  // Ищем только среди строк, которые ещё не были pivot (позиции >= step)
  for (size_t pos = step; pos < n_; ++pos) {
    size_t global_row = row_permutation_[pos];

    // Проверяем, принадлежит ли строка текущему процессу
    if (!IsLocalRow(global_row)) {
      continue;
    }

    size_t local_idx = GlobalToLocal(global_row);
    double value = std::abs(local_matrix_[(local_idx * cols_) + step]);

    if (value > local_max) {
      local_max = value;
      local_row = static_cast<int>(global_row);
    }
  }

  return {local_max, local_row};
}

void SabirovSGaussJordanMethodMPI::NormalizeAndBroadcastPivotRow(size_t global_pivot_row, int pivot_owner,
                                                                 size_t step) {
  if (rank_ == pivot_owner) {
    // Нормализуем pivot строку
    size_t local_idx = GlobalToLocal(global_pivot_row);
    double *row_ptr = &local_matrix_[local_idx * cols_];
    double pivot_value = row_ptr[step];

    for (size_t j = 0; j < cols_; ++j) {
      pivot_row_buf_[j] = row_ptr[j] / pivot_value;
    }

    // Обновляем локальную копию
    for (size_t j = 0; j < cols_; ++j) {
      row_ptr[j] = pivot_row_buf_[j];
    }
  }

  // Рассылаем нормализованную pivot строку всем процессам
  MPI_Bcast(pivot_row_buf_.data(), static_cast<int>(cols_), MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
}

void SabirovSGaussJordanMethodMPI::EliminateColumnLocal(size_t step) {
  size_t pivot_row = row_permutation_[step];

  // Обрабатываем все локальные строки
  for (size_t local_idx = 0; local_idx < local_row_count_; ++local_idx) {
    size_t global_row = local_row_start_ + local_idx;

    // Пропускаем pivot строку
    if (global_row == pivot_row) {
      continue;
    }

    double *row_ptr = &local_matrix_[local_idx * cols_];
    double factor = row_ptr[step];

    if (std::abs(factor) < kEps) {
      row_ptr[step] = 0.0;
      continue;
    }

    for (size_t j = step; j < cols_; ++j) {
      row_ptr[j] -= factor * pivot_row_buf_[j];
    }
    row_ptr[step] = 0.0;
  }
}

int SabirovSGaussJordanMethodMPI::GetRowOwner(size_t global_row) const {
  for (int proc = 0; proc < world_size_; ++proc) {
    size_t start = static_cast<size_t>(row_displs_[proc]);
    size_t end = start + static_cast<size_t>(row_counts_[proc]);
    if (global_row >= start && global_row < end) {
      return proc;
    }
  }
  return 0;
}

size_t SabirovSGaussJordanMethodMPI::GlobalToLocal(size_t global_row) const {
  return global_row - local_row_start_;
}

bool SabirovSGaussJordanMethodMPI::IsLocalRow(size_t global_row) const {
  return global_row >= local_row_start_ && global_row < local_row_start_ + local_row_count_;
}

bool SabirovSGaussJordanMethodMPI::PostProcessingImpl() {
  return true;
}

// ============================================================================
// Однопроцессорная версия (оптимизация для world_size == 1)
// ============================================================================

bool SabirovSGaussJordanMethodMPI::RunSingleProcessGaussJordan() {
  for (size_t i = 0; i < n_; ++i) {
    size_t pivot_row = FindPivotRowSingle(i);
    double max_val = std::abs(local_matrix_[(pivot_row * cols_) + i]);

    if (max_val < kEps) {
      return false;
    }

    if (pivot_row != i) {
      SwapRowsSingle(i, pivot_row);
    }

    NormalizeRowSingle(i, i);
    EliminateColumnSingle(i, i);
  }

  for (size_t i = 0; i < n_; ++i) {
    GetOutput()[i] = local_matrix_[(i * cols_) + n_];
  }

  return true;
}

size_t SabirovSGaussJordanMethodMPI::FindPivotRowSingle(size_t col) const {
  size_t pivot_row = col;
  double max_val = std::abs(local_matrix_[(col * cols_) + col]);

  for (size_t k = col + 1; k < n_; ++k) {
    double val = std::abs(local_matrix_[(k * cols_) + col]);
    if (val > max_val) {
      max_val = val;
      pivot_row = k;
    }
  }

  return pivot_row;
}

void SabirovSGaussJordanMethodMPI::SwapRowsSingle(size_t row1, size_t row2) {
  for (size_t j = 0; j < cols_; ++j) {
    std::swap(local_matrix_[(row1 * cols_) + j], local_matrix_[(row2 * cols_) + j]);
  }
}

void SabirovSGaussJordanMethodMPI::NormalizeRowSingle(size_t row, size_t col) {
  double *row_ptr = &local_matrix_[row * cols_];
  double pivot = row_ptr[col];
  for (size_t j = col; j < cols_; ++j) {
    row_ptr[j] /= pivot;
  }
}

void SabirovSGaussJordanMethodMPI::EliminateColumnSingle(size_t pivot_row, size_t col) {
  double *pivot_ptr = &local_matrix_[pivot_row * cols_];

  for (size_t k = 0; k < n_; ++k) {
    if (k == pivot_row) {
      continue;
    }
    double *other_row = &local_matrix_[k * cols_];
    double factor = other_row[col];
    if (std::abs(factor) < kEps) {
      continue;
    }
    for (size_t j = col; j < cols_; ++j) {
      other_row[j] -= factor * pivot_ptr[j];
    }
    other_row[col] = 0.0;
  }
}

}  // namespace sabirov_s_gauss_jordan_method
