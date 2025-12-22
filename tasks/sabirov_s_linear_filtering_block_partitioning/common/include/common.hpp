#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sabirov_s_linear_filtering_block_partitioning {

// Структура для хранения данных изображения
struct ImageData {
  std::vector<uint8_t> pixels;  // Пиксели в формате RGB
  int width{0};
  int height{0};
  int channels{3};  // RGB = 3 канала

  ImageData() = default;
  ImageData(int w, int h, int ch = 3) : width(w), height(h), channels(ch) {
    pixels.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(ch));
  }

  [[nodiscard]] std::size_t Size() const {
    return pixels.size();
  }

  [[nodiscard]] bool Empty() const {
    return pixels.empty();
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

// Ядро Гаусса 3x3 с использованием std::array
inline constexpr std::array<std::array<float, 3>, 3> kGaussianKernel = {{{1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
                                                                         {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
                                                                         {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F}}};

// Вспомогательная функция для применения ядра Гаусса к одному пикселю
inline float ApplyGaussianKernel(const ImageData &input, int row, int col, int channel) {
  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;

  float sum = 0.0F;

  for (int ky = 0; ky < 3; ky++) {
    for (int kx = 0; kx < 3; kx++) {
      int ny = row + ky - 1;
      int nx = col + kx - 1;

      // Обрабатываем граничные пиксели методом ограничения координат
      ny = std::max(ny, 0);
      ny = std::min(ny, height - 1);
      nx = std::max(nx, 0);
      nx = std::min(nx, width - 1);

      auto input_idx =
          ((static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx)) *
           static_cast<std::size_t>(channels)) +
          static_cast<std::size_t>(channel);
      sum += static_cast<float>(input.pixels[input_idx]) *
             kGaussianKernel.at(static_cast<std::size_t>(ky)).at(static_cast<std::size_t>(kx));
    }
  }

  return sum;
}

}  // namespace sabirov_s_linear_filtering_block_partitioning
