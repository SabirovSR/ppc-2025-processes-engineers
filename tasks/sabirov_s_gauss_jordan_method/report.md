# Метод Гаусса-Жордана

- **Студент:** Сабиров Савелий Русланович, группа 3823Б1ПР1
- **Технология:** MPI (Message Passing Interface) | SEQ (последовательная версия)
- **Вариант:** 17

---

## 1. Введение

Метод Гаусса-Жордана является классическим алгоритмом линейной алгебры для решения систем линейных алгебраических уравнений (СЛАУ). Данный метод представляет собой модификацию метода Гаусса и позволяет привести расширенную матрицу системы к единичной форме за один проход, что упрощает получение решения.

В рамках данной работы реализованы две версии алгоритма:
- **Последовательная (SEQ)** — классическая реализация без использования параллелизма
- **Параллельная (MPI)** — распределённая реализация с использованием технологии MPI для распределения вычислений между несколькими процессами

Целью работы является исследование эффективности параллельной реализации метода Гаусса-Жордана и анализ масштабируемости алгоритма при увеличении числа процессов.

---

## 2. Постановка задачи

### 2.1. Описание задачи

Требуется решить систему линейных алгебраических уравнений вида:

$$A \cdot x = b$$

где:
- $A$ — квадратная матрица коэффициентов размера $n \times n$
- $x$ — вектор неизвестных размера $n$
- $b$ — вектор правых частей размера $n$

Для решения используется метод Гаусса-Жордана с частичным выбором ведущего элемента (partial pivoting) для обеспечения численной устойчивости.

### 2.2. Входные данные

- Расширенная матрица системы $[A|b]$ размера $n \times (n+1)$, представленная в виде одномерного вектора типа `std::vector<double>`
- Элементы матрицы хранятся построчно: сначала $n$ коэффициентов первого уравнения и его правая часть, затем второго и т.д.

### 2.3. Выходные данные

- Вектор решения $x$ размера $n$ типа `std::vector<double>`
- Содержит значения неизвестных $x_1, x_2, \ldots, x_n$

### 2.4. Способ генерации матрицы

Для тестирования используется следующий подход:
1. Генерируется случайный вектор решения $x^*$ с элементами из диапазона $[-10, 10]$
2. Генерируется случайная матрица коэффициентов $A$ с элементами из того же диапазона
3. Вычисляется вектор правых частей $b = A \cdot x^*$
4. Полученная система гарантированно имеет решение $x^*$

Такой подход позволяет проверить корректность алгоритма, сравнивая вычисленное решение с известным.

### 2.5. Пример

Система из трёх уравнений:

$$\begin{cases} 2x_1 + x_2 - x_3 = 8 \\ -3x_1 - x_2 + 2x_3 = -11 \\ -2x_1 + x_2 + 2x_3 = -3 \end{cases}$$

Расширенная матрица: $[A|b] = \begin{pmatrix} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{pmatrix}$

Входной вектор: `{2.0, 1.0, -1.0, 8.0, -3.0, -1.0, 2.0, -11.0, -2.0, 1.0, 2.0, -3.0}`

Решение: $x = (2, 3, -1)^T$

---

## 3. Описание алгоритма

### 3.1. Последовательная версия (SEQ)

Алгоритм метода Гаусса-Жордана с частичным выбором ведущего элемента:

1. **Для каждого столбца $i$ от $0$ до $n-1$:**
   - **Поиск ведущего элемента:** Найти строку $k \geq i$ с максимальным по модулю элементом в столбце $i$
   - **Проверка вырожденности:** Если $|a_{ki}| < \varepsilon$, система вырождена — возврат ошибки
   - **Перестановка строк:** Если $k \neq i$, поменять строки $i$ и $k$ местами
   - **Нормализация ведущей строки:** Разделить все элементы строки $i$ на $a_{ii}$, получая $a_{ii} = 1$
   - **Обнуление столбца:** Для каждой строки $j \neq i$ вычесть из неё строку $i$, умноженную на $a_{ji}$

2. **Извлечение решения:** После приведения матрицы к единичному виду, решение находится в последнем столбце расширенной матрицы

**Сложность алгоритма:** $O(n^3)$ операций умножения и сложения.

---

## 4. Описание схемы параллельного алгоритма

### 4.1. Концепция параллелизации

Параллелизация основана на **блочном распределении строк** матрицы между MPI-процессами. Каждый процесс отвечает за обработку своего блока строк, что позволяет выполнять операции обнуления столбца параллельно.

**Ключевые принципы:**
- Статическое распределение строк (block distribution)
- Минимизация коммуникаций: обмен только ведущей строкой и правыми частями
- Глобальный поиск ведущего элемента через коллективную операцию `MPI_Allreduce`
- Специальный путь для одного процесса — полный обход MPI-накладных расходов

### 4.2. Схема работы параллельного алгоритма

```
┌─────────────────────────────────────────────────────────────────┐
│                    Инициализация                                │
│  • Каждый процесс получает копию матрицы                        │
│  • Вычисляется распределение строк [local_start, local_end)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Для каждого шага i = 0..n-1                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Локальный поиск максимума в столбце i среди своих строк      │
│ 2. MPI_Allreduce(MAXLOC) → глобальный максимум и его владелец   │
│ 3. Обновление логического порядка строк (row_order)             │
│ 4. Владелец нормализует ведущую строку                          │
│ 5. MPI_Bcast → рассылка нормализованной строки всем             │
│ 6. Параллельное обнуление: каждый процесс обрабатывает свои     │
│    строки                                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Сбор результатов                             │
│  • MPI_Allgatherv → сбор правых частей от всех процессов        │
│  • Восстановление решения с учётом перестановок строк           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3. Распределение нагрузки

Строки распределяются **блочно** с учётом остатка от деления:

```
base_rows = n / world_size
remainder = n % world_size

Процесс r владеет строками:
  rows_count = base_rows + (r < remainder ? 1 : 0)
  first_row  = r * base_rows + min(r, remainder)
  last_row   = first_row + rows_count
```

Такое распределение гарантирует разницу не более чем в одну строку между процессами, обеспечивая хорошую балансировку нагрузки.

### 4.4. Коммуникации между процессами

| Операция | Тип | Данные | Объём |
|----------|-----|--------|-------|
| Поиск ведущего элемента | `MPI_Allreduce` (MAXLOC) | (значение, номер строки) | $O(1)$ |
| Рассылка ведущей строки | `MPI_Bcast` | Нормализованная строка | $O(n)$ |
| Сбор решения | `MPI_Allgatherv` | Правые части | $O(n)$ |

**Суммарный объём коммуникаций за весь алгоритм:** $O(n^2)$

---

## 5. Описание MPI-версии (программная реализация)

### 5.1. Архитектура решения

MPI-версия реализована в виде класса `SabirovSGaussJordanMethodMPI`, наследующегося от базового класса `BaseTask`. Архитектура построена на основе паттерна "Pipeline", включающего четыре последовательных этапа:

1. **Validation** — валидация входных данных
2. **PreProcessing** — предварительная обработка
3. **Run** — основные вычисления
4. **PostProcessing** — постобработка результатов

### 5.2. Структура классов

```cpp
namespace sabirov_s_gauss_jordan_method {

// Определение типов
using InType = std::vector<double>;   // Расширенная матрица [A|b]
using OutType = std::vector<double>;  // Вектор решения

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

  bool RunSingleProcessGaussJordan();  // Быстрый путь для 1 процесса
  int GetRowOwner(size_t global_row) const;

  std::vector<double> matrix_;        // Рабочая матрица
  size_t n_{0};                       // Размер системы
  std::vector<size_t> row_displs_;    // Смещения строк для процессов
  std::vector<int> row_counts_int_;   // Количество строк у каждого процесса
  std::vector<int> row_displs_int_;   // Смещения для MPI_Allgatherv
  std::vector<size_t> row_order_;     // Логический порядок строк (перестановки)
  std::vector<size_t> row_position_;  // Позиция строки в row_order
  std::vector<double> pivot_row_buffer_;  // Буфер для ведущей строки
  size_t local_row_start_{0};         // Начало локального блока
  size_t local_row_end_{0};           // Конец локального блока
  int cached_rank_{0};                // Кэшированный ранг процесса
  int cached_world_size_{1};          // Кэшированное число процессов
};

}  // namespace sabirov_s_gauss_jordan_method
```

### 5.3. Реализация методов

#### 5.3.1. Конструктор

```cpp
SabirovSGaussJordanMethodMPI::SabirovSGaussJordanMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}
```

#### 5.3.2. Валидация

```cpp
bool SabirovSGaussJordanMethodMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &input = GetInput();

  if (input.empty()) {
    return false;
  }

  size_t total_size = input.size();

  // Решаем уравнение n*(n+1) = total_size для нахождения n
  double discriminant = 1.0 + 4.0 * static_cast<double>(total_size);
  if (discriminant < 0) {
    return false;
  }

  double n_double = (-1.0 + std::sqrt(discriminant)) / 2.0;
  size_t n = static_cast<size_t>(n_double);

  if (std::abs(n_double - static_cast<double>(n)) > 1e-9 || n == 0) {
    return false;
  }

  if (n * (n + 1) != total_size) {
    return false;
  }

  return true;
}
```

#### 5.3.3. Предварительная обработка

```cpp
bool SabirovSGaussJordanMethodMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &cached_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &cached_world_size_);

  const auto &input = GetInput();
  size_t total_size = input.size();
  double n_double = (-1.0 + std::sqrt(1.0 + 4.0 * static_cast<double>(total_size))) / 2.0;
  n_ = static_cast<size_t>(n_double);

  matrix_.assign(input.begin(), input.end());
  GetOutput().assign(n_, 0.0);

  const size_t cols = n_ + 1;
  pivot_row_buffer_.assign(cols, 0.0);

  // Вычисление блочного распределения строк
  row_displs_.assign(static_cast<size_t>(cached_world_size_) + 1, 0);
  row_counts_int_.assign(cached_world_size_, 0);
  row_displs_int_.assign(cached_world_size_, 0);

  size_t offset = 0;
  size_t base_rows = n_ / static_cast<size_t>(cached_world_size_);
  size_t remainder = n_ % static_cast<size_t>(cached_world_size_);
  for (int r = 0; r < cached_world_size_; ++r) {
    row_displs_[r] = offset;
    size_t rows = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
    row_counts_int_[r] = static_cast<int>(rows);
    row_displs_int_[r] = static_cast<int>(offset);
    offset += rows;
  }
  row_displs_[cached_world_size_] = offset;

  local_row_start_ = row_displs_[cached_rank_];
  local_row_end_ = row_displs_[cached_rank_ + 1];

  // Инициализация массивов для отслеживания перестановок строк
  row_order_.resize(n_);
  row_position_.resize(n_);
  std::iota(row_order_.begin(), row_order_.end(), 0);
  for (size_t i = 0; i < n_; ++i) {
    row_position_[i] = i;
  }

  return true;
}
```

#### 5.3.4. Основные вычисления (MPI-версия)

```cpp
bool SabirovSGaussJordanMethodMPI::RunImpl() {
  const size_t cols = n_ + 1;

  // Быстрый путь для одного процесса — без MPI-накладных расходов
  if (cached_world_size_ == 1) {
    return RunSingleProcessGaussJordan();
  }

  constexpr double kEps = 1e-10;

  for (size_t step = 0; step < n_; ++step) {
    // 1. Локальный поиск максимума в столбце step
    double local_max = 0.0;
    int local_row = -1;

    for (size_t pos = step; pos < n_; ++pos) {
      size_t row_id = row_order_[pos];
      if (row_id < local_row_start_ || row_id >= local_row_end_) {
        continue;
      }
      double value = std::abs(matrix_[row_id * cols + step]);
      if (value > local_max) {
        local_max = value;
        local_row = static_cast<int>(row_id);
      }
    }

    // 2. Глобальный поиск максимума через MPI_Allreduce
    struct { double value; int row; } local_pair{local_max, local_row}, global_pair{};
    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (global_pair.row == -1 || global_pair.value < kEps) {
      return false;  // Вырожденная система
    }

    // 3. Обновление логического порядка строк
    size_t pivot_row_id = static_cast<size_t>(global_pair.row);
    size_t pivot_pos = row_position_[pivot_row_id];
    if (pivot_pos != step) {
      size_t other_row_id = row_order_[step];
      std::swap(row_order_[step], row_order_[pivot_pos]);
      row_position_[pivot_row_id] = step;
      row_position_[other_row_id] = pivot_pos;
    }

    pivot_row_id = row_order_[step];
    const int pivot_owner = GetRowOwner(pivot_row_id);

    // 4. Нормализация ведущей строки владельцем
    if (cached_rank_ == pivot_owner) {
      double *row_ptr = &matrix_[pivot_row_id * cols];
      double pivot_value = row_ptr[step];
      for (size_t j = 0; j < cols; ++j) {
        pivot_row_buffer_[j] = row_ptr[j] / pivot_value;
      }
      std::copy(pivot_row_buffer_.begin(), pivot_row_buffer_.end(), row_ptr);
    }

    // 5. Рассылка нормализованной ведущей строки
    MPI_Bcast(pivot_row_buffer_.data(), static_cast<int>(cols), MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

    if (cached_rank_ != pivot_owner) {
      double *row_ptr = &matrix_[pivot_row_id * cols];
      std::copy(pivot_row_buffer_.begin(), pivot_row_buffer_.end(), row_ptr);
    }

    // 6. Параллельное обнуление столбца — каждый процесс обрабатывает свои строки
    for (size_t pos = 0; pos < n_; ++pos) {
      size_t row_id = row_order_[pos];
      if (row_id == pivot_row_id) continue;
      if (row_id < local_row_start_ || row_id >= local_row_end_) continue;

      double *row_ptr = &matrix_[row_id * cols];
      double factor = row_ptr[step];
      if (std::abs(factor) < kEps) {
        row_ptr[step] = 0.0;
        continue;
      }
      for (size_t j = step; j < cols; ++j) {
        row_ptr[j] -= factor * pivot_row_buffer_[j];
      }
      row_ptr[step] = 0.0;
    }
  }

  // 7. Сбор правых частей от всех процессов
  size_t local_row_count = local_row_end_ - local_row_start_;
  std::vector<double> local_rhs(local_row_count, 0.0);
  for (size_t row = local_row_start_; row < local_row_end_; ++row) {
    local_rhs[row - local_row_start_] = matrix_[row * cols + n_];
  }

  std::vector<double> gathered_rhs(n_, 0.0);
  MPI_Allgatherv(local_rhs.data(), static_cast<int>(local_row_count), MPI_DOUBLE,
                 gathered_rhs.data(), row_counts_int_.data(), row_displs_int_.data(),
                 MPI_DOUBLE, MPI_COMM_WORLD);

  // 8. Восстановление решения с учётом перестановок
  for (size_t step = 0; step < n_; ++step) {
    size_t row_id = row_order_[step];
    GetOutput()[step] = gathered_rhs[row_id];
  }

  return true;
}
```

#### 5.3.5. Постобработка

```cpp
bool SabirovSGaussJordanMethodMPI::PostProcessingImpl() {
  // Постобработка не требуется — результат уже в выходном векторе
  return true;
}
```

Проверяет, что размер выходного вектора равен размеру матрицы `n`.

### 5.4. Преимущества MPI-реализации

1. **Масштабируемость:** Алгоритм корректно работает при любом числе процессов от 1 до n
2. **Минимальные коммуникации:** Обмен только ведущей строкой и правыми частями
3. **Балансировка нагрузки:** Равномерное распределение строк между процессами
4. **Численная устойчивость:** Частичный выбор ведущего элемента с глобальным поиском
5. **Оптимизация для одного процесса:** Специальный путь без MPI-накладных расходов

---

## 6. Экспериментальная установка

### 6.1. Аппаратное обеспечение и операционная система

- **Процессор:** AMD Ryzen 9 9950X3D (16 ядер / 32 потока, 4.30 GHz ~ 5.7 GHz)
- **Оперативная память:** 96 ГБ DDR5 6000 MT/c CL28
- **Операционная система:** Windows 11 Pro 25H2

### 6.2. Инструментарий

- **Компилятор:** MSVC 19.44 (Microsoft Visual C++ Compiler)
- **Стандарт C++:** C++20
- **MPI-реализация:** MS-MPI 10.0 (64-bit)
- **Система сборки:** CMake 4.2.0-rc3
- **Конфигурация сборки:** Release
- **Фреймворк тестирования:** Google Test (из `3rdparty/googletest`)

### 6.3. Переменные окружения

- **`PPC_NUM_PROC`:** количество MPI-процессов, используется при запуске под `mpirun` (значения: 1, 2, 4, 8, 16, 32)
- **`PPC_NUM_THREADS`:** количество потоков (для последовательной версии установлено в 1)
- **`PPC_TEST_TMPDIR`, `PPC_TEST_UID`:** автоматически устанавливаются тестовым фреймворком PPC для изоляции тестов

### 6.4. Генерация и источники данных

- **Размер тестовой матрицы:** 2000 × 2001 (система из 2000 уравнений)
- **Генератор случайных чисел:** Mersenne Twister (std::mt19937) с фиксированным seed=42 для воспроизводимости
- **Диапазон значений:** элементы матрицы и решения из интервала [-10.0, 10.0]
- **Метод проверки:** сравнение вычисленного решения с известным (точность $10^{-6}$)

---

## 7. Результаты и обсуждение

### 7.1. Корректность

Корректность реализации проверена с помощью системы автоматизированных тестов на базе Google Test.

#### 7.1.1. Параметризованные функциональные тесты

Тесты выполняются для систем размерности 3, 5 и 10 с использованием генерации систем с известным решением. Для каждого размера проверяется:
- Совпадение вычисленного решения с известным с точностью $10^{-7}$
- Корректная работа обеих версий (SEQ и MPI)
- Работа MPI-версии при различном числе процессов (2, 4, 8, 16, 32)

#### 7.1.2. Валидационные тесты

Проверяется корректность валидации входных данных:
- Пустой входной вектор → отклонение
- Некорректный размер матрицы (не n×(n+1)) → отклонение
- Нулевой размер системы → отклонение

#### 7.1.3. Тесты граничных случаев

- **Система 1×1:** единственное уравнение `4x = 20` → решение `x = 5`
- **Система с нулевым ведущим элементом:** проверка корректности перестановки строк
- **Вырожденная система:** линейно зависимые строки → возврат ошибки

#### 7.1.4. Результаты тестирования корректности

**Все функциональные тесты успешно пройдены** для обеих реализаций при различном количестве MPI-процессов (1, 2, 4, 8, 16, 32).

**Проверенные свойства:**
- Численная точность решения (относительная погрешность < $10^{-6}$)
- Корректная обработка перестановок строк
- Обнаружение вырожденных систем
- Инвариантность результата относительно числа процессов

### 7.2. Производительность

Для оценки производительности реализованы специализированные тесты с двумя режимами измерений:

1. **`run_task`** — измеряет только время выполнения метода `RunImpl()` (чистые вычисления)
2. **`run_pipeline`** — измеряет время выполнения полного конвейера (Validation + PreProcessing + Run + PostProcessing)

Тестирование проводилось на матрице размера 2000 × 2001 (около 32 МБ данных).

#### 7.2.1. Результаты замеров

| Режим             | Процессов | Время, с   | Ускорение | Эффективность |
|-------------------|-----------|------------|-----------|---------------|
| seq_run_task      | 1         | 0.76959538 | 1.00      | N/A           |
| mpi_run_task      | 2         | 0.88366120 | 0.87      | 43.5%         |
| mpi_run_task      | 4         | 0.46746038 | 1.65      | 41.2%         |
| mpi_run_task      | 8         | 0.26904766 | 2.86      | 35.8%         |
| mpi_run_task      | 16        | 0.29323834 | 2.62      | 16.4%         |
| mpi_run_task      | 32        | 0.55655390 | 1.38      | 4.3%          |
| seq_run_pipeline  | 1         | 3.31669932 | 1.00      | N/A           |
| mpi_run_pipeline  | 2         | 4.19492528 | 0.79      | 39.5%         |
| mpi_run_pipeline  | 4         | 2.14958234 | 1.54      | 38.5%         |
| mpi_run_pipeline  | 8         | 1.16941646 | 2.84      | 35.5%         |
| mpi_run_pipeline  | 16        | 1.12940502 | 2.94      | 18.4%         |
| mpi_run_pipeline  | 32        | 1.51543334 | 2.19      | 6.8%          |

**Формулы расчёта:**
- Ускорение: `Ускорение = Время_Последовательное / Время_Параллельное`
- Эффективность: `Эффективность = (Ускорение / Количество_Процессов) × 100%`

**Анализ результатов на нетривиальных данных:**

1. **Оптимальная точка:** максимальное ускорение достигается при 8 процессах (2.86x для task_run)
2. **Деградация при 16+ процессах:** связана с превышением накладных расходов MPI над полезной работой
3. **Замедление на 2 процессах:** накладные расходы на коммуникации превышают выигрыш от распараллеливания

#### 7.2.2. Влияние на производительность

**Анализ разницы в производительности**

| Фактор | Влияние |
|--------|---------|
| Объём коммуникаций | $O(n^2)$ операций MPI за весь алгоритм |
| Синхронизация | `MPI_Allreduce` на каждом шаге создаёт барьер |
| Размер данных | При n=2000 матрица занимает ~32 МБ |
| Локальность данных | NUMA-эффекты при >8 процессах на одном узле |
| Дисбаланс нагрузки | Минимален благодаря равномерному распределению |

#### 7.2.4. Теоретические ожидания

При идеальном параллелизме ожидается линейное ускорение `S(p) = p`, где `p` — количество процессов.  
Реальное ускорение обычно меньше из-за:

**Факторы, ограничивающие ускорение:**

1. **Закон Амдала:** последовательная часть (поиск ведущего элемента) ограничивает предельное ускорение
2. **Накладные расходы MPI:** инициализация, барьеры синхронизации, копирование буферов
3. **Ширина памяти:** конкуренция за шину памяти при большом числе процессов
4. **NUMA-эффекты:** неоднородный доступ к памяти на многоядерных системах
5. **Гранулярность задач:** при малом объёме работы на процесс накладные расходы доминируют

**Факторы, способствующие масштабируемости:**

1. **Независимость вычислений:** обнуление разных строк не требует синхронизации
2. **Минимальные коммуникации:** только ведущая строка (n+1 элементов) на каждом шаге
3. **Эффективное распределение:** блочная схема обеспечивает локальность данных
4. **Быстрый путь для 1 процесса:** нет MPI-накладных расходов

---

## 8. Заключение

В рамках данной работы реализованы последовательная и параллельная версии метода Гаусса-Жордана для решения систем линейных уравнений.

**Основные результаты работы:**

1. **Реализация и оптимизация алгоритмов:**
   - Разработана последовательная версия (SEQ) метода Гаусса-Жордана с частичным выбором ведущего элемента
   - Реализована параллельная версия (MPI) с блочным распределением строк между процессами
   - Добавлена оптимизация для одного процесса, исключающая MPI-накладные расходы

2. **Корректность:**
   - Все функциональные тесты пройдены успешно
   - Результаты SEQ и MPI версий совпадают с точностью до $10^{-6}$
   - Корректно обрабатываются граничные случаи и вырожденные системы

3. **Производительность и масштабируемость:**
   - Достигнуто ускорение до 2.86x при использовании 8 процессов
   - Оптимальная эффективность (35-43%) наблюдается при 2-8 процессах
   - При >16 процессах эффективность падает из-за накладных расходов MPI

4. **Методологические выводы:**
   - Параллельные алгоритмы линейной алгебры имеют ограниченную масштабируемость на shared-memory системах
   - Накладные расходы на коммуникации становятся критичными при большом числе процессов
   - Для достижения максимальной производительности необходимо учитывать архитектуру системы (NUMA, кэши)

5. **Практическая применимость:**
   - Реализация подходит для решения СЛАУ размерностью до нескольких тысяч уравнений
   - Рекомендуемое число процессов: 4-8 для достижения баланса между скоростью и эффективностью
   - Для больших систем следует рассмотреть гибридный подход (MPI + OpenMP)

**Ограничения:**

- Алгоритм работает только с невырожденными системами
- Эффективность падает при числе процессов, превышающем размерность задачи
- На Windows-системах MS-MPI может иметь ограничения по производительности

**Научная значимость:**

Работа демонстрирует практические аспекты параллелизации классических алгоритмов линейной алгебры с использованием MPI. Полученные результаты подтверждают теоретические ожидания относительно масштабируемости и могут использоваться как референсная реализация для сравнения с другими подходами.

---

## 9. Ссылки

1. Gropp W., Lusk E., Skjellum A. **Using MPI: Portable Parallel Programming with the Message-Passing Interface**. — 3rd ed. — MIT Press, 2014. — 368 p.

2. MPI Forum. **MPI: A Message-Passing Interface Standard. Version 3.1** [Электронный ресурс]. — Режим доступа: https://www.mpi-forum.org/docs/

3. Антонов А.С. **Параллельное программирование с использованием технологии MPI**. — М.: Изд-во МГУ, 2004. — 71 с.

4. Google Test Documentation [Электронный ресурс]. — Режим доступа: https://google.github.io/googletest/

5. Сысоев А. В. Лекции по параллельному программированию. — Н. Новгород: ННГУ, 2025.

---

## Приложение

### Общие определения (common.hpp)

```cpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sabirov_s_gauss_jordan_method {

// Input: расширенная матрица системы [A|b] в виде одномерного вектора
// Размер матрицы n x (n+1), где n - количество уравнений
using InType = std::vector<double>;
// Output: вектор решений системы уравнений
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sabirov_s_gauss_jordan_method
```

### Заголовочный файл последовательной версии (ops_seq.hpp)

```cpp
#pragma once

#include <vector>

#include "sabirov_s_gauss_jordan_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_gauss_jordan_method {

class SabirovSGaussJordanMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SabirovSGaussJordanMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> matrix_;  // Рабочая матрица для метода Гаусса-Жордана
  size_t n_{0};                 // Размер системы (количество уравнений)
};

}  // namespace sabirov_s_gauss_jordan_method
```

### Реализация последовательной версии (ops_seq.cpp)

```cpp
#include "sabirov_s_gauss_jordan_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sabirov_s_gauss_jordan_method {

SabirovSGaussJordanMethodSEQ::SabirovSGaussJordanMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSGaussJordanMethodSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    return false;
  }

  size_t total_size = input.size();
  double discriminant = 1.0 + 4.0 * static_cast<double>(total_size);
  if (discriminant < 0) {
    return false;
  }

  double n_double = (-1.0 + std::sqrt(discriminant)) / 2.0;
  size_t n = static_cast<size_t>(n_double);

  if (std::abs(n_double - static_cast<double>(n)) > 1e-9 || n == 0) {
    return false;
  }

  if (n * (n + 1) != total_size) {
    return false;
  }

  return true;
}

bool SabirovSGaussJordanMethodSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  size_t total_size = input.size();
  double n_double = (-1.0 + std::sqrt(1.0 + 4.0 * static_cast<double>(total_size))) / 2.0;
  size_t n = static_cast<size_t>(n_double);

  matrix_.assign(input.begin(), input.end());
  n_ = n;
  GetOutput().resize(n);

  return true;
}

bool SabirovSGaussJordanMethodSEQ::RunImpl() {
  const size_t n = n_;
  const size_t cols = n + 1;
  constexpr double kEps = 1e-10;

  for (size_t i = 0; i < n; i++) {
    // Поиск ведущего элемента
    size_t pivot_row = i;
    double max_val = std::abs(matrix_[i * cols + i]);

    for (size_t k = i + 1; k < n; k++) {
      double val = std::abs(matrix_[k * cols + i]);
      if (val > max_val) {
        max_val = val;
        pivot_row = k;
      }
    }

    if (max_val < kEps) {
      return false;  // Вырожденная система
    }

    // Перестановка строк
    if (pivot_row != i) {
      for (size_t j = 0; j < cols; j++) {
        std::swap(matrix_[i * cols + j], matrix_[pivot_row * cols + j]);
      }
    }

    // Нормализация ведущей строки
    double *row_ptr = &matrix_[i * cols];
    double pivot = row_ptr[i];
    for (size_t j = i; j < cols; j++) {
      row_ptr[j] /= pivot;
    }

    // Обнуление столбца
    for (size_t k = 0; k < n; k++) {
      if (k == i) continue;
      double *other_row = &matrix_[k * cols];
      double factor = other_row[i];
      if (std::abs(factor) < kEps) {
        other_row[i] = 0.0;
        continue;
      }
      for (size_t j = i; j < cols; j++) {
        other_row[j] -= factor * row_ptr[j];
      }
      other_row[i] = 0.0;
    }
  }

  // Извлечение решения
  for (size_t i = 0; i < n; i++) {
    GetOutput()[i] = matrix_[i * cols + n];
  }

  return true;
}

bool SabirovSGaussJordanMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sabirov_s_gauss_jordan_method
```

### Заголовочный файл MPI-версии (ops_mpi.hpp)

```cpp
#pragma once

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

  bool RunSingleProcessGaussJordan();
  int GetRowOwner(size_t global_row) const;

  std::vector<double> matrix_;
  size_t n_{0};
  std::vector<size_t> row_displs_;
  std::vector<int> row_counts_int_;
  std::vector<int> row_displs_int_;
  std::vector<size_t> row_order_;
  std::vector<size_t> row_position_;
  std::vector<double> pivot_row_buffer_;
  size_t local_row_start_{0};
  size_t local_row_end_{0};
  int cached_rank_{0};
  int cached_world_size_{1};
};

}  // namespace sabirov_s_gauss_jordan_method
```

### Реализация MPI-версии (ops_mpi.cpp)

```cpp
#include "sabirov_s_gauss_jordan_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sabirov_s_gauss_jordan_method {

SabirovSGaussJordanMethodMPI::SabirovSGaussJordanMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SabirovSGaussJordanMethodMPI::ValidationImpl() {
  // ... см. раздел 5.3.2
}

bool SabirovSGaussJordanMethodMPI::PreProcessingImpl() {
  // ... см. раздел 5.3.3
}

bool SabirovSGaussJordanMethodMPI::RunImpl() {
  // ... см. раздел 5.3.4
}

bool SabirovSGaussJordanMethodMPI::PostProcessingImpl() {
  return true;
}

bool SabirovSGaussJordanMethodMPI::RunSingleProcessGaussJordan() {
  // Идентична последовательной версии
  // ... см. ops_seq.cpp
}

int SabirovSGaussJordanMethodMPI::GetRowOwner(size_t global_row) const {
  auto it = std::upper_bound(row_displs_.begin(), row_displs_.end(), global_row);
  int idx = static_cast<int>(std::distance(row_displs_.begin(), it)) - 1;
  return std::clamp(idx, 0, cached_world_size_ - 1);
}

}  // namespace sabirov_s_gauss_jordan_method
```

### Функциональные тесты (обновлённая версия, краткая выжимка)

Полный код функциональных тестов включает:

1. **Параметризованные тесты** — проверка на матрицах размерности 3, 5, 10
2. **Тесты граничных случаев:**
   - `SolvesSingleEquation` — система 1×1
   - `HandlesPivotSwap` — система с нулевым ведущим элементом
   - `DetectsSingularSystem` — вырожденная система

```cpp
TEST(SabirovSGaussJordanMethodEdgeTests, SolvesSingleEquation) {
  const auto input = BuildAugmentedMatrix({{4.0}}, {20.0});
  const OutType expected = {5.0};
  ExpectSolutionForEnabledTasks(input, expected);
}

TEST(SabirovSGaussJordanMethodEdgeTests, HandlesPivotSwap) {
  const auto input = BuildAugmentedMatrix({{0.0, 1.0}, {2.0, 3.0}}, {4.0, 5.0});
  const OutType expected = {-3.5, 4.0};
  ExpectSolutionForEnabledTasks(input, expected);
}

TEST(SabirovSGaussJordanMethodEdgeTests, DetectsSingularSystem) {
  const auto input = BuildAugmentedMatrix({{1.0, 2.0}, {2.0, 4.0}}, {3.0, 6.0});
  ExpectFailureForEnabledTasks(input);
}
```

*Полный код тестов доступен в файле `tests/functional/main.cpp`.*

### Тесты производительности (tests/performance/main.cpp)

```cpp
class SabirovSGaussJordanMethodRunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 2000;  // Размер системы для тестирования
  InType input_data_{};
  OutType expected_solution_{};

  void SetUp() override {
    int n = kCount_;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    expected_solution_.resize(n);
    for (int i = 0; i < n; i++) {
      expected_solution_[i] = dist(gen);
    }

    input_data_.resize(n * (n + 1));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        input_data_[i * (n + 1) + j] = dist(gen);
      }

      double b = 0.0;
      for (int j = 0; j < n; j++) {
        b += input_data_[i * (n + 1) + j] * expected_solution_[j];
      }
      input_data_[i * (n + 1) + n] = b;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_solution_.size()) {
      return false;
    }

    const double tolerance = 1e-6;
    for (size_t i = 0; i < output_data.size(); i++) {
      if (std::abs(output_data[i] - expected_solution_[i]) > tolerance) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabirovSGaussJordanMethodRunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SabirovSGaussJordanMethodMPI, SabirovSGaussJordanMethodSEQ>(
        PPC_SETTINGS_sabirov_s_gauss_jordan_method);

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabirovSGaussJordanMethodRunPerfTest,
                         ppc::util::TupleToGTestValues(kAllPerfTasks),
                         SabirovSGaussJordanMethodRunPerfTest::CustomPerfTestName);
```
