#include <gtest/gtest.h>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"
#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabirov_s_min_val_matrix {

class SabirovSMinValMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
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
