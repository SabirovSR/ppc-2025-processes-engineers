#pragma once

#include <vector>

#include "sabirov_s_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_hypercube {

/// @brief MPI реализация топологии гиперкуба
/// @details Реализует виртуальную топологию гиперкуба без использования
///          MPI_Cart_Create и MPI_Graph_Create. Обеспечивает передачу
///          данных от любого выбранного процесса любому другому процессу.
class SabirovSHypercubeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SabirovSHypercubeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  /// @brief Отправляет данные через гиперкуб по вычисленному маршруту
  /// @param source Ранг процесса-источника
  /// @param dest Ранг процесса-получателя
  /// @param data Данные для передачи
  /// @return Вектор полученных данных (заполнен только на процессе-получателе)
  std::vector<int> SendThroughHypercube(int source, int dest, const std::vector<int> &data);

  /// @brief Находит позицию текущего процесса в маршруте
  /// @param route Маршрут передачи
  /// @return Позиция в маршруте или -1 если процесс не участвует
  [[nodiscard]] int FindRoutePosition(const std::vector<int> &route) const;

  /// @brief Обрабатывает логику отправки для процесса-источника
  /// @param route Маршрут передачи
  /// @param buffer Буфер с данными
  static void ProcessSourceNode(const std::vector<int> &route, const std::vector<int> &buffer);

  /// @brief Обрабатывает логику для промежуточного узла
  /// @param route Маршрут передачи
  /// @param route_position Позиция в маршруте
  /// @param dest Ранг процесса-получателя
  /// @return Полученные данные
  [[nodiscard]] std::vector<int> ProcessIntermediateNode(const std::vector<int> &route, int route_position,
                                                         int dest) const;

  /// @brief Рассылает полученные данные всем процессам
  /// @param dest Ранг процесса-получателя
  void BroadcastReceivedData(int dest);

  /// @brief Рассылает маршрут всем процессам
  void BroadcastRoute();

  int world_size_{0};  ///< Количество процессов в MPI_COMM_WORLD
  int world_rank_{0};  ///< Ранг текущего процесса
  int dimension_{0};   ///< Размерность гиперкуба
};

}  // namespace sabirov_s_hypercube
