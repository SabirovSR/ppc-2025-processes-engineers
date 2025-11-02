#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"

namespace sabirov_s_min_val_matrix {

SabirovSMinValMatrixMPI::SabirovSMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SabirovSMinValMatrixMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SabirovSMinValMatrixMPI::PreProcessingImpl() {
  GetOutput() = 0;
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
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

  // Каждый процесс обрабатывает свои строки
  InType local_sum = 0;

  for (InType i = start_row; i < end_row; i++) {
    std::vector<InType> row(n);
    row[0] = 1;
    for (InType j = 1; j < n; j++) {
      row[j] = (i * n) + j + 1;
    }

    InType min_val = row[0];
    for (InType j = 1; j < n; j++) {
      min_val = std::min(min_val, row[j]);
    }
    local_sum += min_val;
  }

  // Собираем результаты на процессе 0
  InType global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum;
  } else {
    GetOutput() = 0;
  }

  // Рассылаем результат всем процессам
  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  return GetOutput() > 0;
}

bool SabirovSMinValMatrixMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace sabirov_s_min_val_matrix
