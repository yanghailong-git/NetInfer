#include "rms_norm.hpp"
namespace net_infer {

RMSNormLayer::RMSNormLayer() : ParamLayer("rms_norm") {
  // 为单个权重张量（增益/缩放参数）预留空间。
  this->weights_.resize(1);
}

void RMSNormLayer::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << "This function is not implement in the RMSNorm layer！";
}

void RMSNormLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  CHECK(weights.size() == weights_.size());
  this->weights_ = weights;
}

StatusCode RMSNormLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  // 校验输入/输出批次数组。
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the rmsnorm layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the rmsnorm layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the rmsnorm "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  // 校验增益权重张量是否存在。
  if (weights_.empty() || weights_.front()->empty()) {
    LOG(ERROR) << "The weight for the rmsnorm layer is missing.";
    return StatusCode::kInferParamError;
  }

  // 将增益权重包装为 Armadillo 向量以进行逐元素操作。
  std::shared_ptr<Tensor<float>> weight = this->weight(0);
  arma::fvec weight_vec(weight->raw_ptr(), weight->size(), false, true);
  const uint32_t batch_size = inputs.size();

  // 并行处理批次中的每个样本。
#pragma omp parallel for if (batch_size > 1) num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the rmsnorm layer has an "
           "empty tensor "
        << i << " th";

    // 若缺少输出张量，则进行分配。
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    // 确保输入和输出形状匹配。
    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the rmsnorm "
           "layer do not match "
        << i << " th";

    const size_t size = input->size();
    // 将输入和输出数据包装为 Armadillo 向量（非拥有）。
    arma::fvec input_vec(input->raw_ptr(), size, false, true);

    // 计算 RMS 归一化：
    // 1. 对每个元素求平方。
    // 2. 计算平方均值。
    // 3. 计算逆 RMS：1 / sqrt(mean + eps)。
    // 4. 使用可学习的增益权重进行缩放。
    const arma::fvec& input_pow_vec = arma::pow(input_vec, 2.f);
    const float mean_value = arma::mean(input_pow_vec);
    const float norm_value = 1.f / std::sqrt(mean_value + eps_);
    arma::fvec output_vec(output->raw_ptr(), size, false, true);
    output_vec = weight_vec % (norm_value * input_vec);
  }
  return StatusCode::kSuccess;
}
}  // namespace net_infer
