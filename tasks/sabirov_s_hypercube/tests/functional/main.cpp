#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "sabirov_s_hypercube/common/include/common.hpp"
#include "sabirov_s_hypercube/mpi/include/ops_mpi.hpp"
#include "sabirov_s_hypercube/seq/include/ops_seq.hpp"
#include "util/include/util.hpp"

namespace sabirov_s_hypercube {

//============================================================================
// Тестовый класс для последовательной версии
//============================================================================

class SabirovSHypercubeSEQTests : public ::testing::Test {
 protected:
  static void RunTest(int dimension, int source, int dest, const std::vector<int> &data) {
    HypercubeInput input;
    input.dimension = dimension;
    input.source_rank = source;
    input.dest_rank = dest;
    input.data = data;

    auto task = std::make_shared<SabirovSHypercubeSEQ>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    const auto &output = task->GetOutput();

    // Проверяем, что данные переданы корректно
    ASSERT_TRUE(output.success);
    ASSERT_EQ(output.received_data, data);

    // Проверяем корректность маршрута
    ASSERT_FALSE(output.route.empty());
    ASSERT_EQ(output.route.front(), source);
    ASSERT_EQ(output.route.back(), dest);

    // Проверяем, что каждый переход - это переход к соседу
    for (size_t i = 1; i < output.route.size(); ++i) {
      ASSERT_EQ(HammingDistance(output.route[i - 1], output.route[i]), 1);
    }
  }
};

//============================================================================
// Тесты для размерности 1 (2 узла: 0 и 1)
//============================================================================

TEST_F(SabirovSHypercubeSEQTests, Dim1Node0ToNode1) {
  RunTest(1, 0, 1, {42});
}

TEST_F(SabirovSHypercubeSEQTests, Dim1Node1ToNode0) {
  RunTest(1, 1, 0, {100, 200, 300});
}

TEST_F(SabirovSHypercubeSEQTests, Dim1SelfSend) {
  RunTest(1, 0, 0, {1, 2, 3, 4, 5});
}

//============================================================================
// Тесты для размерности 2 (4 узла: 0, 1, 2, 3)
//============================================================================

TEST_F(SabirovSHypercubeSEQTests, Dim2Node0ToNode3) {
  // Узлы 0 и 3 диагонально противоположны
  RunTest(2, 0, 3, {10, 20, 30});
}

TEST_F(SabirovSHypercubeSEQTests, Dim2Node1ToNode2) {
  // Узлы 1 и 2 диагонально противоположны
  RunTest(2, 1, 2, {5, 15, 25, 35});
}

TEST_F(SabirovSHypercubeSEQTests, Dim2NeighborNodes) {
  // Узлы 0 и 1 соседи
  RunTest(2, 0, 1, {7, 14, 21});
}

TEST_F(SabirovSHypercubeSEQTests, Dim2AllPairs) {
  std::vector<int> data = {1, 2, 3};
  for (int src = 0; src < 4; ++src) {
    for (int dst = 0; dst < 4; ++dst) {
      RunTest(2, src, dst, data);
    }
  }
}

//============================================================================
// Тесты для размерности 3 (8 узлов: 0-7)
//============================================================================

TEST_F(SabirovSHypercubeSEQTests, Dim3Node0ToNode7) {
  // Максимальное расстояние (3 хопа)
  RunTest(3, 0, 7, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_F(SabirovSHypercubeSEQTests, Dim3Node3ToNode4) {
  // Также максимальное расстояние
  RunTest(3, 3, 4, {11, 22, 33});
}

TEST_F(SabirovSHypercubeSEQTests, Dim3Node2ToNode6) {
  // Расстояние 1 (соседи в измерении 2)
  RunTest(3, 2, 6, {99});
}

TEST_F(SabirovSHypercubeSEQTests, Dim3LargeData) {
  std::vector<int> large_data(1000);
  for (size_t i = 0; i < large_data.size(); ++i) {
    large_data[i] = static_cast<int>(i);
  }
  RunTest(3, 0, 7, large_data);
}

//============================================================================
// Тесты для размерности 4 (16 узлов: 0-15)
//============================================================================

TEST_F(SabirovSHypercubeSEQTests, Dim4Node0ToNode15) {
  // Максимальное расстояние (4 хопа)
  RunTest(4, 0, 15, {1, 2, 3, 4, 5});
}

TEST_F(SabirovSHypercubeSEQTests, Dim4Node5ToNode10) {
  // Расстояние Хэмминга: 5^10 = 0101^1010 = 1111 = 4
  RunTest(4, 5, 10, {100, 200});
}

TEST_F(SabirovSHypercubeSEQTests, Dim4EmptyData) {
  RunTest(4, 0, 15, {});
}

//============================================================================
// Тесты для размерности 0 (1 узел)
//============================================================================

TEST_F(SabirovSHypercubeSEQTests, Dim0SingleNode) {
  RunTest(0, 0, 0, {42, 43, 44});
}

//============================================================================
// Тесты валидации
//============================================================================

TEST(SabirovSHypercubeValidationTests, SEQInvalidNegativeDimension) {
  HypercubeInput input;
  input.dimension = -1;
  input.source_rank = 0;
  input.dest_rank = 0;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST(SabirovSHypercubeValidationTests, SEQInvalidSourceRank) {
  HypercubeInput input;
  input.dimension = 2;  // 4 узла
  input.source_rank = 5;
  input.dest_rank = 0;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST(SabirovSHypercubeValidationTests, SEQInvalidDestRank) {
  HypercubeInput input;
  input.dimension = 2;  // 4 узла
  input.source_rank = 0;
  input.dest_rank = 10;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST(SabirovSHypercubeValidationTests, SEQNegativeSourceRank) {
  HypercubeInput input;
  input.dimension = 2;
  input.source_rank = -1;
  input.dest_rank = 0;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST(SabirovSHypercubeValidationTests, SEQNegativeDestRank) {
  HypercubeInput input;
  input.dimension = 2;
  input.source_rank = 0;
  input.dest_rank = -1;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

//============================================================================
// Тесты для MPI версии (запускаются под mpirun)
//============================================================================

class SabirovSHypercubeMPITests : public ::testing::Test {
 protected:
  static void RunTest(int dimension, int source, int dest, const std::vector<int> &data) {
    if (!ppc::util::IsUnderMpirun()) {
      GTEST_SKIP() << "Test requires MPI environment";
    }

    HypercubeInput input;
    input.dimension = dimension;
    input.source_rank = source;
    input.dest_rank = dest;
    input.data = data;

    auto task = std::make_shared<SabirovSHypercubeMPI>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    const auto &output = task->GetOutput();

    // Проверяем, что данные переданы корректно
    ASSERT_TRUE(output.success);
    ASSERT_EQ(output.received_data, data);

    // Проверяем корректность маршрута
    ASSERT_FALSE(output.route.empty());
    ASSERT_EQ(output.route.front(), source);
    ASSERT_EQ(output.route.back(), dest);
  }
};

TEST_F(SabirovSHypercubeMPITests, Dim1Node0ToNode1) {
  RunTest(1, 0, 1, {42});
}

TEST_F(SabirovSHypercubeMPITests, Dim1SelfSend) {
  RunTest(1, 0, 0, {1, 2, 3});
}

TEST_F(SabirovSHypercubeMPITests, Dim2Node0ToNode3) {
  RunTest(2, 0, 3, {10, 20, 30});
}

TEST_F(SabirovSHypercubeMPITests, Dim2Node1ToNode2) {
  RunTest(2, 1, 2, {5, 15, 25});
}

TEST_F(SabirovSHypercubeMPITests, Dim3Node0ToNode7) {
  RunTest(3, 0, 7, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_F(SabirovSHypercubeMPITests, Dim3LargeData) {
  std::vector<int> large_data(500);
  for (size_t i = 0; i < large_data.size(); ++i) {
    large_data[i] = static_cast<int>(i);
  }
  RunTest(3, 0, 7, large_data);
}

TEST_F(SabirovSHypercubeMPITests, Dim0SingleNode) {
  RunTest(0, 0, 0, {42});
}

TEST(SabirovSHypercubeValidationTests, MPIInvalidSourceRank) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP() << "Test requires MPI environment";
  }

  HypercubeInput input;
  input.dimension = 2;
  input.source_rank = 5;
  input.dest_rank = 0;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeMPI>(input);
  ASSERT_FALSE(task->Validation());
}

TEST(SabirovSHypercubeValidationTests, MPIInvalidDestRank) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP() << "Test requires MPI environment";
  }

  HypercubeInput input;
  input.dimension = 2;
  input.source_rank = 0;
  input.dest_rank = 10;
  input.data = {1};

  auto task = std::make_shared<SabirovSHypercubeMPI>(input);
  ASSERT_FALSE(task->Validation());
}

//============================================================================
// Тесты вспомогательных функций
//============================================================================

TEST(SabirovSHypercubeUtilTests, IsPowerOfTwo) {
  ASSERT_TRUE(IsPowerOfTwo(1));
  ASSERT_TRUE(IsPowerOfTwo(2));
  ASSERT_TRUE(IsPowerOfTwo(4));
  ASSERT_TRUE(IsPowerOfTwo(8));
  ASSERT_TRUE(IsPowerOfTwo(16));
  ASSERT_TRUE(IsPowerOfTwo(1024));

  ASSERT_FALSE(IsPowerOfTwo(0));
  ASSERT_FALSE(IsPowerOfTwo(3));
  ASSERT_FALSE(IsPowerOfTwo(5));
  ASSERT_FALSE(IsPowerOfTwo(6));
  ASSERT_FALSE(IsPowerOfTwo(7));
  ASSERT_FALSE(IsPowerOfTwo(-1));
}

TEST(SabirovSHypercubeUtilTests, Log2) {
  ASSERT_EQ(Log2(1), 0);
  ASSERT_EQ(Log2(2), 1);
  ASSERT_EQ(Log2(4), 2);
  ASSERT_EQ(Log2(8), 3);
  ASSERT_EQ(Log2(16), 4);
  ASSERT_EQ(Log2(1024), 10);
}

TEST(SabirovSHypercubeUtilTests, GetNeighbor) {
  // Для узла 0 в 3D гиперкубе
  ASSERT_EQ(GetNeighbor(0, 0), 1);  // 000 -> 001
  ASSERT_EQ(GetNeighbor(0, 1), 2);  // 000 -> 010
  ASSERT_EQ(GetNeighbor(0, 2), 4);  // 000 -> 100

  // Для узла 5 (101) в 3D гиперкубе
  ASSERT_EQ(GetNeighbor(5, 0), 4);  // 101 -> 100
  ASSERT_EQ(GetNeighbor(5, 1), 7);  // 101 -> 111
  ASSERT_EQ(GetNeighbor(5, 2), 1);  // 101 -> 001
}

TEST(SabirovSHypercubeUtilTests, HammingDistance) {
  ASSERT_EQ(HammingDistance(0, 0), 0);
  ASSERT_EQ(HammingDistance(0, 1), 1);
  ASSERT_EQ(HammingDistance(0, 7), 3);   // 000 vs 111
  ASSERT_EQ(HammingDistance(5, 10), 4);  // 0101 vs 1010
  ASSERT_EQ(HammingDistance(15, 0), 4);  // 1111 vs 0000
  ASSERT_EQ(HammingDistance(7, 8), 4);   // 0111 vs 1000
}

TEST(SabirovSHypercubeUtilTests, BuildRouteSelfLoop) {
  // Тест маршрута 0 -> 0 (self)
  auto route = BuildRoute(0, 0);
  ASSERT_EQ(route.size(), 1U);
  ASSERT_EQ(route[0], 0);
}

TEST(SabirovSHypercubeUtilTests, BuildRouteSingleHop) {
  // Тест маршрута 0 -> 1 (один хоп)
  auto route = BuildRoute(0, 1);
  ASSERT_EQ(route.size(), 2U);
  ASSERT_EQ(route[0], 0);
  ASSERT_EQ(route[1], 1);
}

TEST(SabirovSHypercubeUtilTests, BuildRouteThreeHops) {
  // Тест маршрута 0 -> 7 (три хопа в 3D)
  auto route = BuildRoute(0, 7);
  ASSERT_EQ(route.size(), 4U);
  ASSERT_EQ(route.front(), 0);
  ASSERT_EQ(route.back(), 7);

  // Проверяем, что каждый переход - это переход к соседу
  for (size_t i = 1; i < route.size(); ++i) {
    ASSERT_EQ(HammingDistance(route[i - 1], route[i]), 1);
  }
}

TEST(SabirovSHypercubeUtilTests, BuildRouteFourHops) {
  // Тест маршрута 5 -> 10 (4 хопа в 4D)
  auto route = BuildRoute(5, 10);
  ASSERT_EQ(route.front(), 5);
  ASSERT_EQ(route.back(), 10);
  ASSERT_EQ(route.size(), static_cast<size_t>(HammingDistance(5, 10) + 1));
}

//============================================================================
// Тесты для сравнения SEQ и MPI версий
//============================================================================

static void CompareSEQandMPI(const HypercubeInput &input) {
  // SEQ версия
  auto task_seq = std::make_shared<SabirovSHypercubeSEQ>(input);
  ASSERT_TRUE(task_seq->Validation());
  ASSERT_TRUE(task_seq->PreProcessing());
  ASSERT_TRUE(task_seq->Run());
  ASSERT_TRUE(task_seq->PostProcessing());

  const auto &seq_output = task_seq->GetOutput();

  // MPI версия (если под mpirun)
  if (!ppc::util::IsUnderMpirun()) {
    return;
  }

  auto task_mpi = std::make_shared<SabirovSHypercubeMPI>(input);
  ASSERT_TRUE(task_mpi->Validation());
  ASSERT_TRUE(task_mpi->PreProcessing());
  ASSERT_TRUE(task_mpi->Run());
  ASSERT_TRUE(task_mpi->PostProcessing());

  const auto &mpi_output = task_mpi->GetOutput();

  // Сравниваем результаты
  ASSERT_EQ(seq_output.success, mpi_output.success);
  ASSERT_EQ(seq_output.received_data, mpi_output.received_data);
  ASSERT_EQ(seq_output.route, mpi_output.route);
}

TEST(SabirovSHypercubeComparisonTests, CompareOutputs) {
  HypercubeInput input;
  input.dimension = 2;
  input.source_rank = 0;
  input.dest_rank = 3;
  input.data = {1, 2, 3, 4, 5};

  CompareSEQandMPI(input);
}

}  // namespace sabirov_s_hypercube
