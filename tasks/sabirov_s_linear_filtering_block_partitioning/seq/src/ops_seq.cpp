#include "sabirov_s_linear_filtering_block_partitioning/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "sabirov_s_linear_filtering_block_partitioning/common/include/common.hpp"

namespace sabirov_s_linear_filtering_block_partitioning {

SabirovSLinearFilteringBlockPartitioningSEQ::SabirovSLinearFilteringBlockPartitioningSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSLinearFilteringBlockPartitioningSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return !input.Empty() && input.width > 0 && input.height > 0 && input.channels == 3;
}

bool SabirovSLinearFilteringBlockPartitioningSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  GetOutput() = ImageData(input.width, input.height, input.channels);
  return !GetOutput().Empty();
}

bool SabirovSLinearFilteringBlockPartitioningSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  if (input.Empty() || input.width < 3 || input.height < 3) {
    return false;
  }

  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;

  // Применяем фильтр Гаусса к каждому пикселю
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      for (int ch = 0; ch < channels; ch++) {
        float sum = ApplyGaussianKernel(input, row, col, ch);
        auto output_idx =
            ((static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col)) *
             static_cast<std::size_t>(channels)) +
            static_cast<std::size_t>(ch);
        output.pixels[output_idx] = static_cast<uint8_t>(std::clamp(sum, 0.0F, 255.0F));
      }
    }
  }

  return true;
}

bool SabirovSLinearFilteringBlockPartitioningSEQ::PostProcessingImpl() {
  return !GetOutput().Empty();
}

}  // namespace sabirov_s_linear_filtering_block_partitioning
