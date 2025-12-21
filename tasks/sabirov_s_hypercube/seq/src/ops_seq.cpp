#include "sabirov_s_hypercube/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "sabirov_s_hypercube/common/include/common.hpp"

namespace sabirov_s_hypercube {

SabirovSHypercubeSEQ::SabirovSHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = HypercubeOutput{.received_data = {}, .route = {}, .success = false};
}

bool SabirovSHypercubeSEQ::ValidationImpl() {
  const auto &input = GetInput();

  // Проверка размерности
  if (input.dimension < 0) {
    return false;
  }

  int num_nodes = 1 << input.dimension;  // 2^dimension

  // Проверка корректности рангов источника и получателя
  if (input.source_rank < 0 || input.source_rank >= num_nodes) {
    return false;
  }

  if (input.dest_rank < 0 || input.dest_rank >= num_nodes) {
    return false;
  }

  return true;
}

bool SabirovSHypercubeSEQ::PreProcessingImpl() {
  dimension_ = GetInput().dimension;
  num_nodes_ = 1 << dimension_;

  // Инициализация выходных данных
  GetOutput().success = false;
  GetOutput().received_data.clear();
  GetOutput().route.clear();

  return true;
}

std::vector<int> SabirovSHypercubeSEQ::EmulateHypercubeTransfer(int source, int dest, const std::vector<int> &data) {
  // Вычисляем маршрут от источника к получателю
  std::vector<int> route = BuildRoute(source, dest);

  // Сохраняем маршрут
  if (!route.empty()) {
    GetOutput().route.assign(route.begin(), route.end());
  } else {
    GetOutput().route.clear();
  }

  // В последовательной версии эмулируем передачу
  // Данные просто копируются, так как нет реальной передачи между процессами
  std::vector<int> result;
  if (!data.empty()) {
    result.assign(data.begin(), data.end());
  }

  // Эмуляция пересылки через промежуточные узлы
  // В реальной системе данные проходили бы через каждый узел маршрута
  // Здесь мы просто проверяем корректность маршрута

  // Проверяем, что каждый шаг маршрута - это переход к соседу в гиперкубе
  for (size_t i = 1; i < route.size(); ++i) {
    int prev = route[i - 1];
    int curr = route[i];

    // Расстояние Хэмминга между соседями должно быть равно 1
    if (HammingDistance(prev, curr) != 1) {
      // Некорректный маршрут
      return {};
    }
  }

  return result;
}

bool SabirovSHypercubeSEQ::RunImpl() {
  const auto &input = GetInput();

  int source = input.source_rank;
  int dest = input.dest_rank;
  const auto &data = input.data;

  // Эмуляция передачи данных через гиперкуб
  std::vector<int> received = EmulateHypercubeTransfer(source, dest, data);

  // Сохраняем результат
  if (!received.empty()) {
    GetOutput().received_data.assign(received.begin(), received.end());
  } else {
    GetOutput().received_data.clear();
  }
  GetOutput().success = true;

  return true;
}

bool SabirovSHypercubeSEQ::PostProcessingImpl() {
  return GetOutput().success;
}

}  // namespace sabirov_s_hypercube
