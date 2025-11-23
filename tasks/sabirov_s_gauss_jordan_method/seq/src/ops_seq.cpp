#include "sabirov_s_gauss_jordan_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"

namespace sabirov_s_gauss_jordan_method {

namespace {
constexpr double kEps = 1e-10;
}  // namespace

SabirovSGaussJordanMethodSEQ::SabirovSGaussJordanMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSGaussJordanMethodSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    return false;
  }

  size_t total_size = input.size();

  double discriminant = 1.0 + (4.0 * static_cast<double>(total_size));
  if (discriminant < 0) {
    return false;
  }

  double n_double = (-1.0 + std::sqrt(discriminant)) / 2.0;
  auto n_val = static_cast<size_t>(n_double);

  if (std::abs(n_double - static_cast<double>(n_val)) > 1e-9 || n_val == 0) {
    return false;
  }

  if (n_val * (n_val + 1) != total_size) {
    return false;
  }

  return true;
}

bool SabirovSGaussJordanMethodSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  size_t total_size = input.size();
  double n_double = (-1.0 + std::sqrt(1.0 + (4.0 * static_cast<double>(total_size)))) / 2.0;
  auto n_val = static_cast<size_t>(n_double);

  matrix_.assign(input.begin(), input.end());
  n_ = n_val;

  GetOutput().resize(n_val);

  return true;
}

size_t SabirovSGaussJordanMethodSEQ::FindPivotRow(size_t col, size_t cols) const {
  size_t pivot_row = col;
  double max_val = std::abs(matrix_[(col * cols) + col]);

  for (size_t k = col + 1; k < n_; k++) {
    double val = std::abs(matrix_[(k * cols) + col]);
    if (val > max_val) {
      max_val = val;
      pivot_row = k;
    }
  }

  return pivot_row;
}

void SabirovSGaussJordanMethodSEQ::SwapRows(size_t row1, size_t row2, size_t cols) {
  for (size_t j = 0; j < cols; j++) {
    std::swap(matrix_[(row1 * cols) + j], matrix_[(row2 * cols) + j]);
  }
}

void SabirovSGaussJordanMethodSEQ::NormalizeRow(size_t row, size_t col, size_t cols) {
  double *row_ptr = &matrix_[row * cols];
  double pivot = row_ptr[col];
  for (size_t j = col; j < cols; j++) {
    row_ptr[j] /= pivot;
  }
}

void SabirovSGaussJordanMethodSEQ::EliminateColumn(size_t pivot_row, size_t col, size_t cols) {
  double *pivot_ptr = &matrix_[pivot_row * cols];

  for (size_t k = 0; k < n_; k++) {
    if (k == pivot_row) {
      continue;
    }
    double *other_row = &matrix_[k * cols];
    double factor = other_row[col];
    if (std::abs(factor) < kEps) {
      other_row[col] = 0.0;
      continue;
    }
    for (size_t j = col; j < cols; j++) {
      other_row[j] -= factor * pivot_ptr[j];
    }
    other_row[col] = 0.0;
  }
}

bool SabirovSGaussJordanMethodSEQ::RunImpl() {
  const size_t n = n_;
  const size_t cols = n + 1;

  for (size_t i = 0; i < n; i++) {
    size_t pivot_row = FindPivotRow(i, cols);
    double max_val = std::abs(matrix_[(pivot_row * cols) + i]);

    if (max_val < kEps) {
      return false;
    }

    if (pivot_row != i) {
      SwapRows(i, pivot_row, cols);
    }

    NormalizeRow(i, i, cols);
    EliminateColumn(i, i, cols);
  }

  for (size_t i = 0; i < n; i++) {
    GetOutput()[i] = matrix_[(i * cols) + n];
  }

  return true;
}

bool SabirovSGaussJordanMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sabirov_s_gauss_jordan_method
