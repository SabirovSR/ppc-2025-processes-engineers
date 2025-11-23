#include "sabirov_s_gauss_jordan_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"

namespace sabirov_s_gauss_jordan_method {

SabirovSGaussJordanMethodSEQ::SabirovSGaussJordanMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSGaussJordanMethodSEQ::ValidationImpl() {
  const auto &input = GetInput();

  // Проверяем, что входные данные не пустые
  if (input.empty()) {
    return false;
  }

  // Вычисляем размер системы (n уравнений, n неизвестных)
  // Размер расширенной матрицы: n x (n+1)
  size_t total_size = input.size();

  // Находим n: n * (n+1) = total_size
  // n^2 + n - total_size = 0
  // n = (-1 + sqrt(1 + 4*total_size)) / 2
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

bool SabirovSGaussJordanMethodSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  size_t total_size = input.size();
  double n_double = (-1.0 + std::sqrt(1.0 + 4.0 * static_cast<double>(total_size))) / 2.0;
  size_t n = static_cast<size_t>(n_double);

  // Копируем входную матрицу для обработки
  matrix_.assign(input.begin(), input.end());
  n_ = n;

  // Инициализируем выходной вектор
  GetOutput().resize(n);

  return true;
}

bool SabirovSGaussJordanMethodSEQ::RunImpl() {
  const size_t n = n_;
  const size_t cols = n + 1;
  constexpr double kEps = 1e-10;

  for (size_t i = 0; i < n; i++) {
    size_t pivot_row = i;
    double max_val = std::abs(matrix_[i * cols + i]);

    for (size_t k = i + 1; k < n; k++) {
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
      for (size_t j = 0; j < cols; j++) {
        std::swap(matrix_[i * cols + j], matrix_[pivot_row * cols + j]);
      }
    }

    double *row_ptr = &matrix_[i * cols];
    double pivot = row_ptr[i];
    for (size_t j = i; j < cols; j++) {
      row_ptr[j] /= pivot;
    }

    for (size_t k = 0; k < n; k++) {
      if (k == i) {
        continue;
      }
      double *other_row = &matrix_[k * cols];
      double factor = other_row[i];
      if (std::abs(factor) < kEps) {
        other_row[i] = 0.0;
        continue;
      }
      for (size_t j = i; j < cols; j++) {
        other_row[j] -= factor * row_ptr[j];
      }
      other_row[i] = 0.0;
    }
  }

  for (size_t i = 0; i < n; i++) {
    GetOutput()[i] = matrix_[i * cols + n];
  }

  return true;
}

bool SabirovSGaussJordanMethodSEQ::PostProcessingImpl() {
  // Постобработка не требуется
  return true;
}

}  // namespace sabirov_s_gauss_jordan_method
