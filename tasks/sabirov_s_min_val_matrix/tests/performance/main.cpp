#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"
#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabirov_s_min_val_matrix {

class SabirovSMinValMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  InType input_data_{};
  std::vector<InType> expected_mins_;

  void SetUp() override {
    input_data_ = kCount_;

    auto generate_value = [](int64_t i, int64_t j) -> InType {
      constexpr int64_t A = 1103515245;
      constexpr int64_t C = 12345;
      constexpr int64_t M = 2147483648;
      int64_t seed = (i * 100000007LL + j * 1000000009LL) ^ 42;
      int64_t val = (A * seed + C) % M;
      return static_cast<InType>((val % 2000001) - 1000000);
    };

    expected_mins_.resize(kCount_);
    for (int i = 0; i < kCount_; i++) {
      InType min_val = generate_value(i, 0);
      for (int j = 1; j < kCount_; j++) {
        InType val = generate_value(i, j);
        min_val = std::min(min_val, val);
      }
      expected_mins_[i] = min_val;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != expected_mins_[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabirovSMinValMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SabirovSMinValMatrixMPI, SabirovSMinValMatrixSEQ>(
    PPC_SETTINGS_sabirov_s_min_val_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SabirovSMinValMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabirovSMinValMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace sabirov_s_min_val_matrix
