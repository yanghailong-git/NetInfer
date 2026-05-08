#include "layer/abstract/layer.hpp"

namespace net_infer {

/**
 * @brief 获取层的权重张量列表（常量引用）
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志。
 */
const std::vector<std::shared_ptr<Tensor<float>>>& Layer<float>::weights() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

/**
 * @brief 获取层的偏置张量列表（常量引用）
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志。
 */
const std::vector<std::shared_ptr<Tensor<float>>>& Layer<float>::bias() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

/**
 * @brief 使用一维浮点向量设置偏置
 * @param bias 偏置数值向量
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志。
 */
void Layer<float>::set_bias(const std::vector<float>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

/**
 * @brief 使用张量列表设置偏置
 * @param bias 偏置张量列表
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志。
 */
void Layer<float>::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

/**
 * @brief 使用一维浮点向量设置权重
 * @param weights 权重数值向量
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志。
 */
void Layer<float>::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

/**
 * @brief 使用张量列表设置权重
 * @param weights 权重张量列表
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志。
 */
void Layer<float>::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

/**
 * @brief 前向推理（带输入输出的重载版本）
 * @param inputs 输入张量列表
 * @param outputs 输出张量列表
 * @return 状态码
 *
 * 基类默认实现：未实现该功能的层调用此接口将触发 FATAL 日志并返回未实现状态。
 */
StatusCode Layer<float>::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  return StatusCode::kFunctionNotImplement;
}

/**
 * @brief 前向推理（从 RuntimeOperator 中获取输入输出的调度版本）
 * @return 状态码
 *
 * 该函数是框架执行推理时的主要调度入口：
 * 1. 从 runtime_operator_ 中获取输入操作数序列
 * 2. 校验输入数据非空
 * 3. 校验输出操作数非空
 * 4. 调用具体层实现的 Forward(inputs, outputs) 完成计算
 */
StatusCode Layer<float>::Forward() {
  LOG_IF(FATAL, this->runtime_operator_.expired()) << "Runtime operator is expired or nullptr";
  const auto& runtime_operator = this->runtime_operator_.lock();

  // 收集该层所有的输入张量数据
  std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
  for (const auto& input_operand_data : runtime_operator->input_operands_seq) {
    if (input_operand_data == nullptr) {
      return StatusCode::kInferInputsEmpty;
    }
    std::copy(input_operand_data->datas.begin(), input_operand_data->datas.end(),
              std::back_inserter(layer_input_datas));
  }

  // 检查输入数据是否为空
  if (layer_input_datas.empty()) {
    LOG(ERROR) << runtime_operator->name << " Layer input data is empty";
    return StatusCode::kInferInputsEmpty;
  }

  // 检查每一个输入张量是否有效
  for (sftensor layer_input_data : layer_input_datas) {
    if (layer_input_data == nullptr || layer_input_data->empty()) {
      LOG(ERROR) << "Layer input data is empty";
      return StatusCode::kInferInputsEmpty;
    }
  }

  // 获取输出操作数并检查有效性
  const std::shared_ptr<RuntimeOperand>& output_operand_datas = runtime_operator->output_operands;
  if (output_operand_datas == nullptr || output_operand_datas->datas.empty()) {
    LOG(ERROR) << "Layer output data is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  // 调用具体层的前向计算
  StatusCode status =
      runtime_operator->layer->Forward(layer_input_datas, output_operand_datas->datas);
  if (status != StatusCode::kSuccess) {
    LOG(ERROR) << "Forward the layer " << runtime_operator->name << " get a error status";
  }
  return status;
}

/**
 * @brief 检查输入输出张量的合法性
 * @param inputs 输入张量列表
 * @param outputs 输出张量列表
 * @return 状态码
 *
 * 基类默认实现：直接返回未实现状态。
 */
StatusCode Layer<float>::Check(const std::vector<sftensor>& inputs,
                               const std::vector<sftensor>& outputs) {
  return StatusCode::kFunctionNotImplement;
}

/**
 * @brief 绑定 RuntimeOperator 到当前层
 * @param runtime_operator 运行时算子对象
 *
 * 该对象包含了层的输入输出操作数以及层本身的配置信息。
 */
void Layer<float>::set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}

}  // namespace net_infer
