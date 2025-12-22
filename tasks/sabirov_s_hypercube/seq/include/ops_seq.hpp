#pragma once

#include <vector>

#include "sabirov_s_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_hypercube {

/// @brief Последовательная реализация топологии гиперкуба
/// @details Эмулирует топологию гиперкуба без использования MPI.
///          Вычисляет маршрут и симулирует передачу данных.
class SabirovSHypercubeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SabirovSHypercubeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  /// @brief Эмулирует передачу данных через гиперкуб
  /// @param source Ранг процесса-источника
  /// @param dest Ранг процесса-получателя
  /// @param data Данные для передачи
  /// @return Вектор полученных данных
  std::vector<int> EmulateHypercubeTransfer(int source, int dest, const std::vector<int> &data);

  int num_nodes_{0};  ///< Количество узлов в гиперкубе (2^dimension)
  int dimension_{0};  ///< Размерность гиперкуба
};

}  // namespace sabirov_s_hypercube
