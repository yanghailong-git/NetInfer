#include "cat.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace net_infer {
CatLayer::CatLayer(int32_t dim) : NonParamLayer("cat"), dim_(dim) {}

StatusCode CatLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                             std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  StatusCode status_code = Check(inputs, outputs);
  if (status_code != StatusCode::kSuccess) {
    return status_code;
  }

  const uint32_t output_size = outputs.size();
  const uint32_t packet_size = inputs.size() / output_size;
#pragma omp parallel for num_threads(outputs.size())
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    uint32_t copy_channel_offset = 0;
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    for (uint32_t j = i; j < inputs.size(); j += output_size) {
      const std::shared_ptr<Tensor<float>>& input = inputs.at(j);
      const uint32_t in_rows = input->rows();
      const uint32_t in_cols = input->cols();
      const uint32_t in_channels = input->channels();

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(in_channels * packet_size, in_rows, in_cols);
        outputs.at(i) = output;
      }
      CHECK(output->channels() == in_channels * packet_size && output->rows() == in_rows &&
            output->cols() == in_cols)
          << "The output tensor array in the cat layer "
             "has an incorrectly sized tensor "
          << i << " th";

      const uint32_t plane_size = in_rows * in_cols;
      memcpy(output->raw_ptr(copy_channel_offset * plane_size), input->raw_ptr(),
             sizeof(float) * plane_size * in_channels);
      copy_channel_offset += input->channels();
    }
  }
  return StatusCode::kSuccess;
}

StatusCode CatLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                    std::shared_ptr<Layer<float>>& cat_layer) {
  if (!op) {
    LOG(ERROR) << "The cat operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the cat layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParamError;
  }

  auto dim_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParamError;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return StatusCode::kSuccess;
}

StatusCode CatLayer::Check(const std::vector<sftensor>& inputs,
                           const std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  for (const auto& input_data : inputs) {
    if (input_data == nullptr || inputs.empty()) {
      return StatusCode::kInferInputsEmpty;
    }
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the cat layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension parameter of cat layer is error";
    return StatusCode::kInferParamError;
  }

  const uint32_t output_size = outputs.size();
  if (inputs.size() % output_size != 0) {
    LOG(ERROR) << "The input and output tensor array size of cat layer do not match";
    return StatusCode::kInferDimMismatch;
  }
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kCatCreateInstance(CatLayer::CreateInstance, "torch.cat");
}  // namespace net_infer