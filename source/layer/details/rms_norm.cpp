#include "rms_norm.hpp"
namespace net_infer {

RMSNormLayer::RMSNormLayer() : ParamLayer("rms_norm") {
  // Reserve space for exactly one weight tensor (the gain/scale parameter).
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
  // Validate input/output batch arrays.
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

  // Validate presence of the gain weight tensor.
  if (weights_.empty() || weights_.front()->empty()) {
    LOG(ERROR) << "The weight for the rmsnorm layer is missing.";
    return StatusCode::kInferParamError;
  }

  // Wrap the gain weight into an Armadillo vector for element-wise operations.
  std::shared_ptr<Tensor<float>> weight = this->weight(0);
  arma::fvec weight_vec(weight->raw_ptr(), weight->size(), false, true);
  const uint32_t batch_size = inputs.size();

  // Process each sample in the batch in parallel.
#pragma omp parallel for if (batch_size > 1) num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the rmsnorm layer has an "
           "empty tensor "
        << i << " th";

    // Allocate output tensor if missing.
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    // Ensure input and output shapes match.
    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the rmsnorm "
           "layer do not match "
        << i << " th";

    const size_t size = input->size();
    // Wrap input and output data as Armadillo vectors (non-owning).
    arma::fvec input_vec(input->raw_ptr(), size, false, true);

    // Compute RMS normalization:
    // 1. Square each element.
    // 2. Compute mean of squares.
    // 3. Compute inverse RMS: 1 / sqrt(mean + eps).
    // 4. Scale by the learnable gain weight.
    const arma::fvec& input_pow_vec = arma::pow(input_vec, 2.f);
    const float mean_value = arma::mean(input_pow_vec);
    const float norm_value = 1.f / std::sqrt(mean_value + eps_);
    arma::fvec output_vec(output->raw_ptr(), size, false, true);
    output_vec = weight_vec % (norm_value * input_vec);
  }
  return StatusCode::kSuccess;
}
}  // namespace net_infer
