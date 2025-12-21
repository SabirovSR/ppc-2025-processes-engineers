#include "sabirov_s_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "sabirov_s_hypercube/common/include/common.hpp"

namespace sabirov_s_hypercube {

SabirovSHypercubeMPI::SabirovSHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = HypercubeOutput{.received_data = {}, .route = {}, .success = false};
}

bool SabirovSHypercubeMPI::ValidationImpl() {
  const auto &input = GetInput();

  // Проверка размерности
  if (input.dimension < 0) {
    return false;
  }

  int num_nodes = 1 << input.dimension;  // 2^dimension

  // Проверка, что количество процессов соответствует топологии гиперкуба
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size < num_nodes) {
    return false;
  }

  // Проверка корректности рангов источника и получателя
  if (input.source_rank < 0 || input.source_rank >= num_nodes) {
    return false;
  }

  if (input.dest_rank < 0 || input.dest_rank >= num_nodes) {
    return false;
  }

  return true;
}

bool SabirovSHypercubeMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

  dimension_ = GetInput().dimension;

  // Прогрев MPI коммуникаций - обмен данными для инициализации буферов
  // и соединений. Это критически важно для стабильного измерения времени,
  // т.к. первые MPI операции обычно медленнее ("холодный старт").
  // Используем те же типы операций, что и в основном алгоритме.
  int warmup_data = world_rank_;
  MPI_Bcast(&warmup_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // Инициализация выходных данных
  GetOutput().success = false;
  GetOutput().received_data.clear();
  GetOutput().route.clear();

  return true;
}

int SabirovSHypercubeMPI::FindRoutePosition(const std::vector<int> &route) const {
  for (size_t i = 0; i < route.size(); ++i) {
    if (route[i] == world_rank_) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void SabirovSHypercubeMPI::ProcessSourceNode(const std::vector<int> &route, const std::vector<int> &buffer) {
  if (route.size() <= 1) {
    return;
  }

  int next_node = route[1];
  auto data_size = static_cast<int>(buffer.size());
  MPI_Send(&data_size, 1, MPI_INT, next_node, 0, MPI_COMM_WORLD);

  if (data_size > 0 && !buffer.empty()) {
    MPI_Send(buffer.data(), data_size, MPI_INT, next_node, 1, MPI_COMM_WORLD);
  }
}

[[nodiscard]] std::vector<int> SabirovSHypercubeMPI::ProcessIntermediateNode(const std::vector<int> &route,
                                                                             int route_position, int dest) const {
  std::vector<int> buffer;
  int prev_node = route[route_position - 1];

  // Получаем данные от предыдущего узла
  int data_size = 0;
  MPI_Recv(&data_size, 1, MPI_INT, prev_node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (data_size > 0) {
    buffer.resize(data_size);
    if (!buffer.empty()) {
      MPI_Recv(buffer.data(), data_size, MPI_INT, prev_node, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Если это не конечный узел, передаём дальше
  if (route_position < static_cast<int>(route.size()) - 1) {
    int next_node = route[route_position + 1];
    MPI_Send(&data_size, 1, MPI_INT, next_node, 0, MPI_COMM_WORLD);

    if (data_size > 0 && !buffer.empty()) {
      MPI_Send(buffer.data(), data_size, MPI_INT, next_node, 1, MPI_COMM_WORLD);
    }
  }

  // Если это получатель, возвращаем данные
  if (world_rank_ == dest) {
    return buffer;
  }

  return {};
}

void SabirovSHypercubeMPI::BroadcastReceivedData(int dest) {
  // Рассылаем результат всем процессам для корректной проверки
  int success_int = 0;
  if (world_rank_ == dest) {
    success_int = GetOutput().success ? 1 : 0;
  }
  MPI_Bcast(&success_int, 1, MPI_INT, dest, MPI_COMM_WORLD);
  GetOutput().success = (success_int == 1);

  // Рассылаем полученные данные всем процессам
  int recv_size = 0;
  if (world_rank_ == dest) {
    recv_size = static_cast<int>(GetOutput().received_data.size());
  }
  MPI_Bcast(&recv_size, 1, MPI_INT, dest, MPI_COMM_WORLD);

  if (recv_size > 0) {
    if (world_rank_ != dest) {
      GetOutput().received_data.resize(recv_size);
    }
    if (!GetOutput().received_data.empty()) {
      MPI_Bcast(GetOutput().received_data.data(), recv_size, MPI_INT, dest, MPI_COMM_WORLD);
    }
  }
}

void SabirovSHypercubeMPI::BroadcastRoute() {
  auto route_size = static_cast<int>(GetOutput().route.size());
  MPI_Bcast(&route_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    GetOutput().route.resize(route_size);
  }
  if (route_size > 0 && !GetOutput().route.empty()) {
    MPI_Bcast(GetOutput().route.data(), route_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

std::vector<int> SabirovSHypercubeMPI::SendThroughHypercube(int source, int dest, const std::vector<int> &data) {
  std::vector<int> result;
  std::vector<int> route = BuildRoute(source, dest);
  GetOutput().route = route;

  // Если маршрут состоит только из одного узла (source == dest)
  if (route.size() == 1) {
    if (world_rank_ == source) {
      if (!data.empty()) {
        result.assign(data.begin(), data.end());
      }
    }
    return result;
  }

  int num_nodes = 1 << dimension_;
  int route_position = FindRoutePosition(route);

  // Если процесс не участвует в маршруте или его ранг >= num_nodes, пропускаем
  if (route_position == -1 || world_rank_ >= num_nodes) {
    MPI_Barrier(MPI_COMM_WORLD);
    return result;
  }

  // Обрабатываем в зависимости от роли в маршруте
  if (world_rank_ == source) {
    SabirovSHypercubeMPI::ProcessSourceNode(route, data);
  } else if (route_position > 0) {
    result = ProcessIntermediateNode(route, route_position, dest);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return result;
}

bool SabirovSHypercubeMPI::RunImpl() {
  // Синхронизация всех процессов перед началом передачи данных.
  // Это критически важно для корректного измерения времени при
  // многократных вызовах Run() без PreProcessing между ними (режим TaskRun).
  MPI_Barrier(MPI_COMM_WORLD);

  const auto &input = GetInput();

  int source = input.source_rank;
  int dest = input.dest_rank;
  const auto &data = input.data;

  // Передача данных через гиперкуб
  std::vector<int> received = SendThroughHypercube(source, dest, data);

  // Получатель сохраняет данные
  if (world_rank_ == dest) {
    if (!received.empty()) {
      GetOutput().received_data.assign(received.begin(), received.end());
    } else {
      GetOutput().received_data.clear();
    }
    GetOutput().success = true;
  }

  // Рассылаем результаты всем процессам
  BroadcastReceivedData(dest);
  BroadcastRoute();

  return GetOutput().success;
}

bool SabirovSHypercubeMPI::PostProcessingImpl() {
  return GetOutput().success;
}

}  // namespace sabirov_s_hypercube
