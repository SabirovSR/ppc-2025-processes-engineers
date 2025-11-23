#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"
#include "sabirov_s_gauss_jordan_method/mpi/include/ops_mpi.hpp"
#include "sabirov_s_gauss_jordan_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabirov_s_gauss_jordan_method {

class SabirovSGaussJordanMethodRunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 2000;
  InType input_data_;
  OutType expected_solution_;

  void SetUp() override {
    int n = kCount_;

    // Генерируем систему линейных уравнений с известным решением
    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    // Создаем известное решение
    expected_solution_.resize(n);
    for (int i = 0; i < n; i++) {
      expected_solution_[i] = dist(gen);
    }

    // Создаем коэффициенты системы и вычисляем правую часть
    input_data_.resize(static_cast<size_t>(n) * static_cast<size_t>(n + 1));
    for (int i = 0; i < n; i++) {
      // Генерируем коэффициенты
      for (int j = 0; j < n; j++) {
        input_data_[(static_cast<size_t>(i) * static_cast<size_t>(n + 1)) + static_cast<size_t>(j)] = dist(gen);
      }

      // Вычисляем правую часть
      double b = 0.0;
      for (int j = 0; j < n; j++) {
        b += input_data_[(static_cast<size_t>(i) * static_cast<size_t>(n + 1)) + static_cast<size_t>(j)] *
             expected_solution_[j];
      }
      input_data_[(static_cast<size_t>(i) * static_cast<size_t>(n + 1)) + static_cast<size_t>(n)] = b;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что размер решения корректен
    if (output_data.size() != expected_solution_.size()) {
      return false;
    }

    // Проверяем, что решение близко к ожидаемому
    const double tolerance = 1e-6;
    for (size_t i = 0; i < output_data.size(); i++) {
      if (std::abs(output_data[i] - expected_solution_[i]) > tolerance) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabirovSGaussJordanMethodRunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SabirovSGaussJordanMethodMPI, SabirovSGaussJordanMethodSEQ>(
        PPC_SETTINGS_sabirov_s_gauss_jordan_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SabirovSGaussJordanMethodRunPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabirovSGaussJordanMethodRunPerfTest, kGtestValues, kPerfTestName);

}  // namespace sabirov_s_gauss_jordan_method
