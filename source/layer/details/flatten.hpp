#ifndef NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

// Flatten 层：通过将 start_dim 到 end_dim 的维度展平来重塑张量
// 对应 PyTorch 的 torch.flatten 操作
class FlattenLayer : public NonParamLayer {
 public:
  // 构造函数：start_dim 和 end_dim 定义要展平的维度范围
  explicit FlattenLayer(int32_t start_dim, int32_t end_dim);

  // 前向传播：展平每个输入张量的指定维度范围
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 工厂方法：从 RuntimeOperator 创建 FlattenLayer
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& flatten_layer);

 private:
  int32_t start_dim_ = 0;  // 展平的起始维度
  int32_t end_dim_ = 0;    // 展平的结束维度（包含）
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
