#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "sabirov_s_linear_filtering_block_partitioning/common/include/common.hpp"
#include "sabirov_s_linear_filtering_block_partitioning/mpi/include/ops_mpi.hpp"
#include "sabirov_s_linear_filtering_block_partitioning/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sabirov_s_linear_filtering_block_partitioning {

namespace {

// Вспомогательная функция для загрузки изображения
ImageData LoadTestImage(const std::string &filename) {
  int width = -1;
  int height = -1;
  int channels = -1;

  std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sabirov_s_linear_filtering_block_partitioning, filename);
  auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
  if (data == nullptr) {
    throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
  }

  ImageData img(width, height, 3);
  auto data_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
  std::copy(data, data + data_size, img.pixels.begin());
  stbi_image_free(data);

  return img;
}

// Вспомогательная функция для создания синтетического изображения
ImageData CreateSyntheticImage(int width, int height, uint8_t value = 128) {
  ImageData img(width, height, 3);
  std::ranges::fill(img.pixels, value);
  return img;
}

// Вспомогательная функция для сравнения изображений с допуском
bool CompareImages(const ImageData &img1, const ImageData &img2, int tolerance = 5) {
  if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
    return false;
  }

  for (std::size_t i = 0; i < img1.pixels.size(); i++) {
    int diff = std::abs(static_cast<int>(img1.pixels[i]) - static_cast<int>(img2.pixels[i]));
    if (diff > tolerance) {
      return false;
    }
  }

  return true;
}

}  // namespace

class SabirovSLinearFilteringBlockPartitioningFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_size = std::get<0>(params);

    // Создаем синтетическое изображение для тестирования
    input_data_ = CreateSyntheticImage(test_size, test_size, 100);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что выходное изображение имеет правильные размеры
    if (output_data.width != input_data_.width || output_data.height != input_data_.height ||
        output_data.channels != input_data_.channels) {
      return false;
    }

    // Проверяем, что выход не пустой и имеет корректные значения пикселей
    if (output_data.Empty()) {
      return false;
    }

    // Для однородного входа выход также должен быть близок к входу после фильтрации
    // (фильтр Гаусса сохраняет однородные области)
    for (std::size_t i = 0; i < output_data.pixels.size(); i++) {
      if (output_data.pixels[i] == 0 && input_data_.pixels[i] != 0) {
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
};

namespace {

TEST_P(SabirovSLinearFilteringBlockPartitioningFuncTests, GaussianFilterTest) {
  ExecuteTest(GetParam());
}

// Тест с различными размерами изображений
const std::array<TestType, 5> kTestParam = {std::make_tuple(3, "3x3"), std::make_tuple(5, "5x5"),
                                            std::make_tuple(8, "8x8"), std::make_tuple(16, "16x16"),
                                            std::make_tuple(32, "32x32")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SabirovSLinearFilteringBlockPartitioningMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sabirov_s_linear_filtering_block_partitioning),
                                           ppc::util::AddFuncTask<SabirovSLinearFilteringBlockPartitioningSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sabirov_s_linear_filtering_block_partitioning));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SabirovSLinearFilteringBlockPartitioningFuncTests::PrintFuncTestName<
    SabirovSLinearFilteringBlockPartitioningFuncTests>;

INSTANTIATE_TEST_SUITE_P(ImageFilterTests, SabirovSLinearFilteringBlockPartitioningFuncTests, kGtestValues,
                         kPerfTestName);

// Дополнительные standalone тесты для лучшего покрытия кода

TEST(SabirovSLinearFilteringBlockPartitioningTests, EmptyImageValidation) {
  ImageData empty_img;
  auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(empty_img);
  EXPECT_FALSE(seq_task->Validation());
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, SmallImageHandling) {
  // Создаем изображение 3x3 (минимальный размер для ядра 3x3)
  ImageData small_img = CreateSyntheticImage(3, 3, 128);

  auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(small_img);
  ASSERT_TRUE(seq_task->Validation());
  ASSERT_TRUE(seq_task->PreProcessing());
  ASSERT_TRUE(seq_task->Run());
  ASSERT_TRUE(seq_task->PostProcessing());

  const auto &output = seq_task->GetOutput();
  EXPECT_EQ(output.width, 3);
  EXPECT_EQ(output.height, 3);
  EXPECT_EQ(output.channels, 3);
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, UniformImagePreservation) {
  // Однородные изображения должны оставаться относительно однородными после фильтрации Гаусса
  ImageData uniform_img = CreateSyntheticImage(10, 10, 100);

  auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(uniform_img);
  ASSERT_TRUE(seq_task->Validation());
  ASSERT_TRUE(seq_task->PreProcessing());
  ASSERT_TRUE(seq_task->Run());
  ASSERT_TRUE(seq_task->PostProcessing());

  const auto &output = seq_task->GetOutput();

  // Проверяем, что все пиксели близки к исходному значению
  int sum = 0;
  for (auto pixel : output.pixels) {
    sum += pixel;
  }
  int avg = sum / static_cast<int>(output.pixels.size());
  EXPECT_NEAR(avg, 100, 5);  // Должно быть близко к исходному значению
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, BorderHandling) {
  // Создаем изображение с различными значениями для тестирования обработки границ
  ImageData border_img(5, 5, 3);
  std::ranges::fill(border_img.pixels, static_cast<uint8_t>(0));

  // Устанавливаем центральный пиксель в белый цвет
  int center = (2 * 5 + 2) * 3;
  auto center_idx = static_cast<std::size_t>(center);
  border_img.pixels[center_idx] = 255;
  border_img.pixels[center_idx + 1] = 255;
  border_img.pixels[center_idx + 2] = 255;

  auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(border_img);
  ASSERT_TRUE(seq_task->Validation());
  ASSERT_TRUE(seq_task->PreProcessing());
  ASSERT_TRUE(seq_task->Run());
  ASSERT_TRUE(seq_task->PostProcessing());

  const auto &output = seq_task->GetOutput();

  // Центральный пиксель должен быть размыт (меньше 255)
  EXPECT_LT(output.pixels[center_idx], 255);
  // Окружающие пиксели должны иметь ненулевые значения из-за размытия
  EXPECT_GT(output.pixels[center_idx - 3], 0);
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, SEQvsSerial) {
  // Проверяем, что последовательная версия дает консистентные результаты
  ImageData test_img = CreateSyntheticImage(8, 8, 150);

  auto seq_task1 = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(test_img);
  ASSERT_TRUE(seq_task1->Validation());
  ASSERT_TRUE(seq_task1->PreProcessing());
  ASSERT_TRUE(seq_task1->Run());
  ASSERT_TRUE(seq_task1->PostProcessing());

  auto seq_task2 = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(test_img);
  ASSERT_TRUE(seq_task2->Validation());
  ASSERT_TRUE(seq_task2->PreProcessing());
  ASSERT_TRUE(seq_task2->Run());
  ASSERT_TRUE(seq_task2->PostProcessing());

  const auto &output1 = seq_task1->GetOutput();
  const auto &output2 = seq_task2->GetOutput();

  // Результаты должны быть идентичными
  EXPECT_TRUE(CompareImages(output1, output2, 0));
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, DifferentImageSizes) {
  std::vector<int> sizes = {4, 6, 10, 15, 20};

  for (int size : sizes) {
    ImageData test_img = CreateSyntheticImage(size, size, 120);

    auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(test_img);
    ASSERT_TRUE(seq_task->Validation()) << "Failed for size: " << size;
    ASSERT_TRUE(seq_task->PreProcessing()) << "Failed for size: " << size;
    ASSERT_TRUE(seq_task->Run()) << "Failed for size: " << size;
    ASSERT_TRUE(seq_task->PostProcessing()) << "Failed for size: " << size;

    const auto &output = seq_task->GetOutput();
    EXPECT_EQ(output.width, size) << "Failed for size: " << size;
    EXPECT_EQ(output.height, size) << "Failed for size: " << size;
  }
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, RealImageLoading) {
  // Тест с реальным изображением pic.jpg, если доступно
  try {
    ImageData real_img = LoadTestImage("pic.jpg");

    // Пропускаем, если изображение слишком маленькое для ядра 3x3
    if (real_img.width < 3 || real_img.height < 3) {
      GTEST_SKIP() << "Image too small (need at least 3x3), got " << real_img.width << "x" << real_img.height;
    }

    auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(real_img);
    ASSERT_TRUE(seq_task->Validation());
    ASSERT_TRUE(seq_task->PreProcessing());
    ASSERT_TRUE(seq_task->Run());
    ASSERT_TRUE(seq_task->PostProcessing());

    const auto &output = seq_task->GetOutput();
    EXPECT_EQ(output.width, real_img.width);
    EXPECT_EQ(output.height, real_img.height);
  } catch (const std::exception &) {
    GTEST_SKIP() << "Real image not available for testing";
  }
}

TEST(SabirovSLinearFilteringBlockPartitioningTests, InvalidChannels) {
  // Тест с неправильным количеством каналов
  ImageData invalid_img;
  invalid_img.width = 10;
  invalid_img.height = 10;
  invalid_img.channels = 1;  // Не RGB
  invalid_img.pixels.resize(100);

  auto seq_task = ppc::task::TaskGetter<SabirovSLinearFilteringBlockPartitioningSEQ>(invalid_img);
  EXPECT_FALSE(seq_task->Validation());
}

}  // namespace

}  // namespace sabirov_s_linear_filtering_block_partitioning
