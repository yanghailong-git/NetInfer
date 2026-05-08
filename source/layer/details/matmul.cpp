#include "matmul.hpp"
namespace net_infer {

LLamaMatmulLayer::LLamaMatmulLayer(int32_t weight_dim0, int32_t weight_dim1)
    : ParamLayer("matmul"), weight_dim0_(weight_dim0), weight_dim1_(weight_dim1) {
  // Reserve space for exactly one weight tensor.
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
  // Validate input/output batch arrays.
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

  // Validate weight tensor.
  if (this->weights_.empty()) {
    LOG(ERROR) << "The weight tensor in the matmul layer is empty";
    return StatusCode::kInferParamError;
  }

  if (weights_.size() != 1) {
    LOG(ERROR) << "Need one weight tensor in the matmul layer";
    return StatusCode::kInferParamError;
  }

  // w @ x: perform matrix multiplication for each sample in the batch.
  uint32_t batch = inputs.size();
#pragma omp parallel for if (batch > 1) num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    std::shared_ptr<Tensor<float>> input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the matmul layer has an empty tensor " << i << " th";
    const std::vector<uint32_t>& input_shapes = input->raw_shapes();
    CHECK(!input_shapes.empty() && input_shapes.size() <= 2);

    // Determine input dimensions: support both 1-D and 2-D inputs.
    uint32_t input_dim0 = 1;
    uint32_t input_dim1 = 1;
    if (input_shapes.size() == 1) {
      input_dim0 = input_shapes.front();
    } else {
      input_dim0 = input_shapes.at(0);
      input_dim1 = input_shapes.at(1);
    }
    // input_dim0 must match the weight's inner dimension.
    CHECK_EQ(input_dim0, weight_dim1_);

    // Wrap input data into an Armadillo matrix (transposed view).
    // input_vec has shape (input_dim1, input_dim0).
    arma::fmat input_vec(input->raw_ptr(), input_dim1, input_dim0, false, true);
    const std::shared_ptr<Tensor<float>>& weight = weights_.front();
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      // Create output tensor if not already allocated.
      output = std::make_shared<Tensor<float>>(1, input_dim1, weight_dim0_);
      outputs.at(i) = output;
    }

    // Verify output tensor shape.
    const auto& output_raw_shapes = output->shapes();
    if (output_raw_shapes.size() == 3) {
      CHECK(output_raw_shapes.at(1) == weight_dim0_ && output_raw_shapes.at(2) == input_dim1)
          << "The row of output tensor should be same to input dim 1 and the "
             "col of output tensor should be same to weight dim 0.";
    } else {
      LOG(FATAL) << "The shape of output tensor need be equal to one or two";
    }

    // Choose computation path based on input dimensionality.
    if (input_dim1 == 1) {
      // Special case: input is effectively a vector.
      float* output_ptr = output->raw_ptr();
      float* weight_ptr = weight->raw_ptr();
#pragma omp parallel for
      for (int32_t j = 0; j < weight_dim0_; ++j) {
        // Extract j-th column (row in memory) of weight as a sub-vector.
        arma::fmat sub_weight(weight_ptr + j * weight_dim1_, weight_dim1_, 1, false, true);
        *(output_ptr + j) = arma::as_scalar(input_vec * sub_weight);
      }
    } else {
      // General case: full matrix multiplication.
      // weight_data has shape (weight_dim1_, weight_dim0_) — transposed storage.
      arma::fmat weight_data(weight->raw_ptr(), weight_dim1_, weight_dim0_, false, true);
      if (weight_dim0_ == 1) {
        // Output is a single column.
        arma::fmat output_mat(output->raw_ptr(), input_dim1, weight_dim0_, false, true);
        output_mat = input_vec * weight_data;
      } else {
        // Output shape is (weight_dim0_, input_dim1) after transpose.
        arma::fmat output_mat(output->raw_ptr(), weight_dim0_, input_dim1, false, true);
        output_mat = (input_vec * weight_data).t();
      }
    }
  }
  return StatusCode::kSuccess;
}
}  // namespace net_infer
