#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"
#include "sabirov_s_gauss_jordan_method/mpi/include/ops_mpi.hpp"
#include "sabirov_s_gauss_jordan_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sabirov_s_gauss_jordan_method {

namespace {

constexpr double kCheckTolerance = 1e-7;

InType BuildAugmentedMatrix(const std::vector<std::vector<double>> &a, const std::vector<double> &b) {
  if (a.empty() || a.size() != b.size()) {
    return {};
  }
  const size_t n = a.size();
  InType input(n * (n + 1), 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      input[(i * (n + 1)) + j] = a[i][j];
    }
    input[(i * (n + 1)) + n] = b[i];
  }
  return input;
}

template <typename TaskType>
OutType ExecuteTaskWithInput(const InType &input) {
  auto task = ppc::task::TaskGetter<TaskType, InType>(input);
  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_TRUE(task->Run());
  EXPECT_TRUE(task->PostProcessing());
  return task->GetOutput();
}

template <typename TaskType>
void ExpectRunFailure(const InType &input) {
  auto task = ppc::task::TaskGetter<TaskType, InType>(input);
  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_FALSE(task->Run());
}

void ExpectSolutionForEnabledTasks(const InType &input, const OutType &expected) {
  auto seq_result = ExecuteTaskWithInput<SabirovSGaussJordanMethodSEQ>(input);
  ASSERT_EQ(seq_result.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(seq_result[i], expected[i], kCheckTolerance);
  }

  if (ppc::util::IsUnderMpirun()) {
    auto mpi_result = ExecuteTaskWithInput<SabirovSGaussJordanMethodMPI>(input);
    ASSERT_EQ(mpi_result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_NEAR(mpi_result[i], expected[i], kCheckTolerance);
    }
  }
}

void ExpectFailureForEnabledTasks(const InType &input) {
  ExpectRunFailure<SabirovSGaussJordanMethodSEQ>(input);
  if (ppc::util::IsUnderMpirun()) {
    ExpectRunFailure<SabirovSGaussJordanMethodMPI>(input);
  }
}

}  // namespace

class SabirovSGaussJordanMethodRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int n = std::get<0>(params);

    // Генерируем систему линейных уравнений с известным решением
    std::mt19937 gen(42);
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

      // Вычисляем правую часть b[i] = A[i][0]*x[0] + A[i][1]*x[1] + ... + A[i][n-1]*x[n-1]
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

 private:
  InType input_data_;
  OutType expected_solution_;
};

namespace {

TEST_P(SabirovSGaussJordanMethodRunFuncTests, GaussJordanSolver) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"),
                                            std::make_tuple(10, "10")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SabirovSGaussJordanMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sabirov_s_gauss_jordan_method),
                                           ppc::util::AddFuncTask<SabirovSGaussJordanMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sabirov_s_gauss_jordan_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SabirovSGaussJordanMethodRunFuncTests::PrintFuncTestName<SabirovSGaussJordanMethodRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussJordanTests, SabirovSGaussJordanMethodRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

TEST(SabirovSGaussJordanMethodEdgeTests, SolvesSingleEquation) {
  const auto input = BuildAugmentedMatrix({{4.0}}, {20.0});
  const OutType expected = {5.0};
  ExpectSolutionForEnabledTasks(input, expected);
}

TEST(SabirovSGaussJordanMethodEdgeTests, HandlesPivotSwap) {
  const auto input = BuildAugmentedMatrix({{0.0, 1.0}, {2.0, 3.0}}, {4.0, 5.0});
  const OutType expected = {-3.5, 4.0};
  ExpectSolutionForEnabledTasks(input, expected);
}

TEST(SabirovSGaussJordanMethodEdgeTests, DetectsSingularSystem) {
  const auto input = BuildAugmentedMatrix({{1.0, 2.0}, {2.0, 4.0}}, {3.0, 6.0});
  ExpectFailureForEnabledTasks(input);
}

}  // namespace sabirov_s_gauss_jordan_method
