#pragma once

#include "sabirov_s_linear_filtering_block_partitioning/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_linear_filtering_block_partitioning {

class SabirovSLinearFilteringBlockPartitioningMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SabirovSLinearFilteringBlockPartitioningMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sabirov_s_linear_filtering_block_partitioning
