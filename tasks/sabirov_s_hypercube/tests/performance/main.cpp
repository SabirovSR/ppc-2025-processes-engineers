#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "sabirov_s_hypercube/common/include/common.hpp"
#include "sabirov_s_hypercube/mpi/include/ops_mpi.hpp"
#include "sabirov_s_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabirov_s_hypercube {

class SabirovSHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // Размер данных для передачи
  static constexpr int kDataSize = 10000000;

  InType input_data_{};

  void SetUp() override {
    // Вычисляем размерность на основе количества MPI процессов
    int world_size = 1;
#ifdef PPC_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
    // Находим максимальную размерность, для которой 2^dimension <= world_size
    int dimension = 0;
    while ((1 << (dimension + 1)) <= world_size) {
      dimension++;
    }

    // Настраиваем входные данные для теста производительности
    input_data_.dimension = dimension;
    input_data_.source_rank = 0;
    input_data_.dest_rank = (1 << dimension) - 1;  // Последний узел (максимальное расстояние)

    // Генерируем тестовые данные
    input_data_.data.resize(kDataSize);
    for (size_t i = 0; i < input_data_.data.size(); ++i) {
      input_data_.data[i] = static_cast<int>(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что передача прошла успешно
    if (!output_data.success) {
      return false;
    }

    // Проверяем, что данные получены корректно
    if (output_data.received_data.size() != input_data_.data.size()) {
      return false;
    }

    // Проверяем содержимое данных
    return output_data.received_data == input_data_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabirovSHypercubePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SabirovSHypercubeMPI, SabirovSHypercubeSEQ>(PPC_SETTINGS_sabirov_s_hypercube);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SabirovSHypercubePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabirovSHypercubePerfTests, kGtestValues, kPerfTestName);

}  // namespace sabirov_s_hypercube
