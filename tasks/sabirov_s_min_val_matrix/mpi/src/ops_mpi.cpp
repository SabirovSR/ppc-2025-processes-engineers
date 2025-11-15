#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"

namespace sabirov_s_min_val_matrix {

SabirovSMinValMatrixMPI::SabirovSMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SabirovSMinValMatrixMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput().empty());
}

bool SabirovSMinValMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool SabirovSMinValMatrixMPI::RunImpl() {
  InType n = GetInput();
  if (n == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Распределяем строки между процессами
  int rows_per_proc = n / size;
  int remainder = n % size;

  int start_row = (rank * rows_per_proc) + std::min(rank, remainder);
  int num_local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
  int end_row = start_row + num_local_rows;

  // Каждый процесс находит минимумы для своих строк
  std::vector<InType> local_mins;
  local_mins.reserve(num_local_rows);

  for (InType i = start_row; i < end_row; i++) {
    std::vector<InType> row(n);
    row[0] = 1;
    for (InType j = 1; j < n; j++) {
      row[j] = (i * n) + j + 1;
    }

    // Нахождение минимума в строке
    InType min_val = row[0];
    for (InType j = 1; j < n; j++) {
      min_val = std::min(min_val, row[j]);
    }
    local_mins.push_back(min_val);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
    recvcounts[i] = proc_rows;
    displs[i] = (i * rows_per_proc) + std::min(i, remainder);
  }

  GetOutput().resize(n);

  // Собираем результаты на процессе 0
  if (rank == 0) {
    MPI_Gatherv(local_mins.data(), num_local_rows, MPI_INT, GetOutput().data(), recvcounts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_mins.data(), num_local_rows, MPI_INT, nullptr, recvcounts.data(), displs.data(), MPI_INT, 0,
                MPI_COMM_WORLD);
  }

  // Рассылаем результат всем процессам
  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

bool SabirovSMinValMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(GetInput()));
}

}  // namespace sabirov_s_min_val_matrix
