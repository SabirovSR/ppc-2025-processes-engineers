#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "sabirov_s_linear_filtering_block_partitioning/common/include/common.hpp"
#include "sabirov_s_linear_filtering_block_partitioning/mpi/include/ops_mpi.hpp"
#include "sabirov_s_linear_filtering_block_partitioning/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabirov_s_linear_filtering_block_partitioning {

namespace {
// Вспомогательная функция для создания тестового изображения
ImageData CreatePerfTestImage(int size, uint8_t value = 128) {
  ImageData img(size, size, 3);
  std::ranges::fill(img.pixels, value);
  return img;
}
}  // namespace

class SabirovSLinearFilteringBlockPartitioningPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // Используем большое изображение для тестов производительности (512x512 хорошо подходит)
  const int kImageSize_ = 512;
  InType input_data_;

  void SetUp() override {
    input_data_ = CreatePerfTestImage(kImageSize_, 150);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что размеры выхода соответствуют входу
    if (output_data.width != input_data_.width || output_data.height != input_data_.height ||
        output_data.channels != input_data_.channels) {
      return false;
    }

    // Проверяем, что выход не пустой и имеет корректный размер
    return !output_data.Empty() && output_data.pixels.size() == input_data_.pixels.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabirovSLinearFilteringBlockPartitioningPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SabirovSLinearFilteringBlockPartitioningMPI,
                                                       SabirovSLinearFilteringBlockPartitioningSEQ>(
    PPC_SETTINGS_sabirov_s_linear_filtering_block_partitioning);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SabirovSLinearFilteringBlockPartitioningPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabirovSLinearFilteringBlockPartitioningPerfTests, kGtestValues, kPerfTestName);

}  // namespace sabirov_s_linear_filtering_block_partitioning
