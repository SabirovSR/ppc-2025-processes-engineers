#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sabirov_s_hypercube {

/// @brief Структура входных данных для топологии гиперкуба
struct HypercubeInput {
  int dimension{0};       ///< Размерность гиперкуба (n), количество процессов = 2^n
  int source_rank{0};     ///< Ранг процесса-источника
  int dest_rank{0};       ///< Ранг процесса-получателя
  std::vector<int> data;  ///< Данные для передачи
};

/// @brief Структура выходных данных для топологии гиперкуба
struct HypercubeOutput {
  std::vector<int> received_data;  ///< Полученные данные на процессе-получателе
  std::vector<int> route;          ///< Маршрут передачи (последовательность рангов)
  bool success;                    ///< Флаг успешности передачи
};

using InType = HypercubeInput;
using OutType = HypercubeOutput;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

/// @brief Проверяет, является ли число степенью двойки
/// @param n Число для проверки
/// @return true если n = 2^k для некоторого k >= 0
inline bool IsPowerOfTwo(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

/// @brief Вычисляет логарифм по основанию 2 (размерность гиперкуба)
/// @param n Количество узлов (должно быть степенью двойки)
/// @return Размерность гиперкуба
inline int Log2(int n) {
  int result = 0;
  while (n > 1) {
    n >>= 1;
    result++;
  }
  return result;
}

/// @brief Получает соседа узла в заданном измерении
/// @param rank Ранг узла
/// @param dimension Номер измерения (0 до n-1)
/// @return Ранг соседа
inline int GetNeighbor(int rank, int dimension) {
  return rank ^ (1 << dimension);
}

/// @brief Вычисляет расстояние Хэмминга между двумя узлами
/// @param a Первый узел
/// @param b Второй узел
/// @return Количество различающихся бит (минимальное расстояние в гиперкубе)
inline int HammingDistance(int a, int b) {
  int diff = a ^ b;
  int count = 0;
  while (diff != 0) {
    count += diff & 1;
    diff >>= 1;
  }
  return count;
}

/// @brief Строит маршрут от источника к получателю в гиперкубе
/// @param source Ранг источника
/// @param dest Ранг получателя
/// @return Вектор рангов, представляющий маршрут (включая source и dest)
inline std::vector<int> BuildRoute(int source, int dest) {
  std::vector<int> route;
  route.push_back(source);

  int current = source;
  int diff = source ^ dest;

  // Идем по биту за раз, исправляя отличающиеся биты
  int bit = 0;
  while (diff != 0) {
    if ((diff & 1) != 0) {
      current = GetNeighbor(current, bit);
      route.push_back(current);
    }
    diff >>= 1;
    bit++;
  }

  return route;
}

}  // namespace sabirov_s_hypercube
