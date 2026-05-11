#include "matmul.hpp"
namespace net_infer {

LLamaMatmulLayer::LLamaMatmulLayer(int32_t weight_dim0, int32_t weight_dim1)
    : ParamLayer("matmul"), weight_dim0_(weight_dim0), weight_dim1_(weight_dim1) {
  // 为单个权重张量预留空间。
  this->weights_.resize(1);
}

void LLamaMatmulLayer::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << "This function is not implement in the LLamaMatmul layer！";
}

void LLamaMatmulLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  CHECK(weights.size() == weights_.size());
  this->weights_ = weights;
}

StatusCode LLamaMatmulLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  // 校验输入/输出批次数组。
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the matmul layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the matmul layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the matmul "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  // 校验权重张量。
  if (this->weights_.empty()) {
    LOG(ERROR) << "The weight tensor in the matmul layer is empty";
    return StatusCode::kInferParamError;
  }

  if (weights_.size() != 1) {
    LOG(ERROR) << "Need one weight tensor in the matmul layer";
    return StatusCode::kInferParamError;
  }

  // w @ x：为批次中的每个样本执行矩阵乘法。
  uint32_t batch = inputs.size();
#pragma omp parallel for if (batch > 1) num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    std::shared_ptr<Tensor<float>> input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the matmul layer has an empty tensor " << i << " th";
    const std::vector<uint32_t>& input_shapes = input->raw_shapes();
    CHECK(!input_shapes.empty() && input_shapes.size() <= 2);

    // 确定输入维度：支持一维和二维输入。
    uint32_t input_dim0 = 1;
    uint32_t input_dim1 = 1;
    if (input_shapes.size() == 1) {
      input_dim0 = input_shapes.front();
    } else {
      input_dim0 = input_shapes.at(0);
      input_dim1 = input_shapes.at(1);
    }
    // input_dim0 必须与权重的内部维度匹配。
    CHECK_EQ(input_dim0, weight_dim1_);

    // 将输入数据包装为 Armadillo 矩阵（转置视图）。
    // input_vec 的形状为 (input_dim1, input_dim0)。
    arma::fmat input_vec(input->raw_ptr(), input_dim1, input_dim0, false, true);
    const std::shared_ptr<Tensor<float>>& weight = weights_.front();
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      // 若输出张量尚未分配，则进行创建。
      output = std::make_shared<Tensor<float>>(1, input_dim1, weight_dim0_);
      outputs.at(i) = output;
    }

    // 校验输出张量形状。
    const auto& output_raw_shapes = output->shapes();
    if (output_raw_shapes.size() == 3) {
      CHECK(output_raw_shapes.at(1) == weight_dim0_ && output_raw_shapes.at(2) == input_dim1)
          << "The row of output tensor should be same to input dim 1 and the "
             "col of output tensor should be same to weight dim 0.";
    } else {
      LOG(FATAL) << "The shape of output tensor need be equal to one or two";
    }

    // 根据输入维度选择计算路径。
    if (input_dim1 == 1) {
      // 特殊情况：输入实际上为向量。
      float* output_ptr = output->raw_ptr();
      float* weight_ptr = weight->raw_ptr();
#pragma omp parallel for
      for (int32_t j = 0; j < weight_dim0_; ++j) {
        // 将权重的第 j 列（内存中的行）提取为子向量。
        arma::fmat sub_weight(weight_ptr + j * weight_dim1_, weight_dim1_, 1, false, true);
        *(output_ptr + j) = arma::as_scalar(input_vec * sub_weight);
      }
    } else {
      // 一般情况：完整矩阵乘法。
      // weight_data 的形状为 (weight_dim1_, weight_dim0_) — 转置存储。
      arma::fmat weight_data(weight->raw_ptr(), weight_dim1_, weight_dim0_, false, true);
      if (weight_dim0_ == 1) {
        // 输出为单列。
        arma::fmat output_mat(output->raw_ptr(), input_dim1, weight_dim0_, false, true);
        output_mat = input_vec * weight_data;
      } else {
        // 转置后的输出形状为 (weight_dim0_, input_dim1)。
        arma::fmat output_mat(output->raw_ptr(), weight_dim0_, input_dim1, false, true);
        output_mat = (input_vec * weight_data).t();
      }
    }
  }
  return StatusCode::kSuccess;
}
}  // namespace net_infer
