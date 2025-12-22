#include "sabirov_s_linear_filtering_block_partitioning/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "sabirov_s_linear_filtering_block_partitioning/common/include/common.hpp"

namespace sabirov_s_linear_filtering_block_partitioning {

SabirovSLinearFilteringBlockPartitioningMPI::SabirovSLinearFilteringBlockPartitioningMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSLinearFilteringBlockPartitioningMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    bool valid = !input.Empty() && input.width > 0 && input.height > 0 && input.channels == 3;
    int valid_flag = valid ? 1 : 0;
    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid;
  }

  int valid_flag = 0;
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag == 1;
}

bool SabirovSLinearFilteringBlockPartitioningMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    const auto &input = GetInput();
    GetOutput() = ImageData(input.width, input.height, input.channels);

    // Рассылаем размеры изображения всем процессам
    std::array<int, 3> dims = {input.width, input.height, input.channels};
    MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    std::array<int, 3> dims = {0, 0, 0};
    MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    GetInput() = ImageData(dims[0], dims[1], dims[2]);
    GetOutput() = ImageData(dims[0], dims[1], dims[2]);
  }

  return !GetOutput().Empty();
}

namespace {
// Вспомогательная функция для фильтрации блока строк
void FilterRows(const ImageData &input, ImageData &output, int start_row, int end_row) {
  const int width = input.width;
  const int channels = input.channels;

  for (int row = start_row; row < end_row; row++) {
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
}

// Вспомогательная функция для рассылки изображения
void BroadcastImage(ImageData &input, int rank, int size) {
  if (rank == 0) {
    for (int dest = 1; dest < size; dest++) {
      MPI_Send(input.pixels.data(), static_cast<int>(input.pixels.size()), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(input.pixels.data(), static_cast<int>(input.pixels.size()), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

// Вспомогательная функция для сбора результатов
void GatherResults(ImageData &output, int rank, int size, int rows_per_proc, int remainder, int width, int channels) {
  if (rank == 0) {
    for (int src = 1; src < size; src++) {
      int src_start = (src * rows_per_proc) + std::min(src, remainder);
      int src_end = src_start + rows_per_proc + (src < remainder ? 1 : 0);
      int src_size = (src_end - src_start) * width * channels;

      if (src_size > 0) {
        auto offset =
            static_cast<std::size_t>(src_start) * static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
        MPI_Recv(output.pixels.data() + offset, src_size, MPI_UNSIGNED_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    int start_row = (rank * rows_per_proc) + std::min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_size = (end_row - start_row) * width * channels;

    if (local_size > 0) {
      auto offset =
          static_cast<std::size_t>(start_row) * static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
      MPI_Send(output.pixels.data() + offset, local_size, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
    }
  }
}
}  // namespace

bool SabirovSLinearFilteringBlockPartitioningMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input = GetInput();
  auto &output = GetOutput();

  if (input.width < 3 || input.height < 3) {
    return false;
  }

  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;

  // Рассылаем входное изображение от процесса 0 всем процессам
  BroadcastImage(input, rank, size);

  // Вычисляем блочное разбиение (распределяем строки между процессами)
  int rows_per_proc = height / size;
  int remainder = height % size;

  int start_row = (rank * rows_per_proc) + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

  // Каждый процесс фильтрует свои назначенные строки
  FilterRows(input, output, start_row, end_row);

  // Собираем результаты на процессе 0
  GatherResults(output, rank, size, rows_per_proc, remainder, width, channels);

  // Рассылаем полный результат от процесса 0 всем процессам
  MPI_Bcast(output.pixels.data(), static_cast<int>(output.pixels.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool SabirovSLinearFilteringBlockPartitioningMPI::PostProcessingImpl() {
  return !GetOutput().Empty();
}

}  // namespace sabirov_s_linear_filtering_block_partitioning
