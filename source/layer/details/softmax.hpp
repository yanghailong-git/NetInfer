#ifndef NET_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#define NET_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace net_infer {
// Softmax层：沿指定维度计算softmax概率分布
class SoftmaxLayer : public NonParamLayer {
 public:
  // 构造函数，dim为softmax计算的维度，默认为-1（最后一个维度）
  explicit SoftmaxLayer(int32_t dim = -1);

  // 前向传播，对输入tensor沿dim维度计算softmax
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 根据运行时算子参数创建SoftmaxLayer实例
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& softmax_layer);

 private:
  // softmax运算的维度
  int32_t softmax_dim_ = -1;
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_SOFTMAX_HPP_
