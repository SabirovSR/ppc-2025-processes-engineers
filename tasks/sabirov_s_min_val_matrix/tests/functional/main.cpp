#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"
#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sabirov_s_min_val_matrix {

class SabirovSMinValMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;
    std::vector<uint8_t> img;
    // Read image
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_example_processes, "pic.jpg");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 0);
      if (data == nullptr) {
        throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
      }
      img = std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width * height * channels)));
      stbi_image_free(data);
      if (std::cmp_not_equal(width, height)) {
        throw std::runtime_error("width != height: ");
      }
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = width - height + std::min(std::accumulate(img.begin(), img.end(), 0), channels);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }
    return std::ranges::all_of(output_data, [](const auto &val) { return val == 1; });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SabirovSMinValMatrixFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SabirovSMinValMatrixMPI, InType>(kTestParam, PPC_SETTINGS_sabirov_s_min_val_matrix),
    ppc::util::AddFuncTask<SabirovSMinValMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_sabirov_s_min_val_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SabirovSMinValMatrixFuncTests::PrintFuncTestName<SabirovSMinValMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SabirovSMinValMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

namespace {

class SabirovSMinValMatrixAdditionalTests : public ::testing::Test {
 protected:
  static void ValidateOutput(const OutType &output, InType n) {
    ASSERT_EQ(output.size(), static_cast<size_t>(n));
    ASSERT_TRUE(std::ranges::all_of(output, [](const auto &val) { return val == 1; }));
  }

  static void RunTest(InType n) {
    auto task_seq = std::make_shared<SabirovSMinValMatrixSEQ>(n);
    ASSERT_TRUE(task_seq->Validation());
    ASSERT_TRUE(task_seq->PreProcessing());
    ASSERT_TRUE(task_seq->Run());
    ASSERT_TRUE(task_seq->PostProcessing());
    ValidateOutput(task_seq->GetOutput(), n);
  }
};

TEST_F(SabirovSMinValMatrixAdditionalTests, MinimalMatrix1x1) {
  RunTest(1);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, SmallMatrix2x2) {
  RunTest(2);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, MediumMatrix4x4) {
  RunTest(4);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, Matrix6x6) {
  RunTest(6);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, LargeMatrix8x8) {
  RunTest(8);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, LargeMatrix10x10) {
  RunTest(10);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, VeryLargeMatrix15x15) {
  RunTest(15);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, VeryLargeMatrix20x20) {
  RunTest(20);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, ExtraLargeMatrix25x25) {
  RunTest(25);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, OddSizeMatrix9x9) {
  RunTest(9);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, OddSizeMatrix11x11) {
  RunTest(11);
}

TEST_F(SabirovSMinValMatrixAdditionalTests, OddSizeMatrix13x13) {
  RunTest(13);
}

}  // namespace

}  // namespace sabirov_s_min_val_matrix
