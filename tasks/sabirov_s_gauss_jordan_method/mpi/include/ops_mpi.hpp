#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_gauss_jordan_method {

class SabirovSGaussJordanMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SabirovSGaussJordanMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Распределение данных от ранга 0
  void DistributeMatrixFromRoot();

  // Сбор результатов на ранге 0
  void GatherResultsToRoot();

  // Основной MPI алгоритм Гаусса-Жордана
  bool RunDistributedGaussJordan();

  // Поиск локального pivot элемента
  [[nodiscard]] std::pair<double, int> FindLocalPivot(size_t step) const;

  // Нормализация и рассылка pivot строки
  void NormalizeAndBroadcastPivotRow(size_t global_pivot_row, int pivot_owner, size_t step);

  // Элиминация столбца на локальных строках
  void EliminateColumnLocal(size_t step);

  // Получение владельца глобальной строки
  [[nodiscard]] int GetRowOwner(size_t global_row) const;

  // Преобразование глобального индекса строки в локальный
  [[nodiscard]] size_t GlobalToLocal(size_t global_row) const;

  // Проверка, принадлежит ли строка текущему процессу
  [[nodiscard]] bool IsLocalRow(size_t global_row) const;

  // Однопроцессорная версия (оптимизация для world_size == 1)
  bool RunSingleProcessGaussJordan();
  [[nodiscard]] size_t FindPivotRowSingle(size_t col) const;
  void SwapRowsSingle(size_t row1, size_t row2);
  void NormalizeRowSingle(size_t row, size_t col);
  void EliminateColumnSingle(size_t pivot_row, size_t col);

  // Размерности задачи
  size_t n_{0};     // Количество уравнений/неизвестных
  size_t cols_{0};  // Количество столбцов (n + 1)

  // Данные матрицы
  std::vector<double> full_matrix_;    // Полная матрица (только на ранге 0 до распределения)
  std::vector<double> local_matrix_;   // Локальные строки матрицы
  std::vector<double> pivot_row_buf_;  // Буфер для pivot строки

  // Распределение строк по процессам
  size_t local_row_count_{0};     // Количество строк на текущем процессе
  size_t local_row_start_{0};     // Глобальный индекс первой локальной строки
  std::vector<int> row_counts_;   // Количество строк на каждом процессе
  std::vector<int> row_displs_;   // Смещения строк для каждого процесса
  std::vector<int> send_counts_;  // Количество элементов для Scatterv/Gatherv
  std::vector<int> send_displs_;  // Смещения элементов для Scatterv/Gatherv

  // Отображение логических позиций на глобальные индексы строк (для pivot swap)
  std::vector<size_t> row_permutation_;  // row_permutation_[logical_pos] = global_row_id

  // MPI контекст
  int rank_{0};
  int world_size_{1};
};

}  // namespace sabirov_s_gauss_jordan_method
