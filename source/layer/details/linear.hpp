#ifndef NET_INFER_SOURCE_LAYER_LINEAR_HPP_
#define NET_INFER_SOURCE_LAYER_LINEAR_HPP_
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace net_infer {

// Linear（全连接）层
// 计算线性变换：y = xW^T + b
// 对应 PyTorch 的 nn.Linear
class LinearLayer : public ParamLayer {
 public:
  //  explicit LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim, uint32_t out_dim,
  //  bool use_bias = true);

  // 构造函数：根据输入/输出特征维度初始化线性层
  explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

  // 校验输入、输出、权重和偏置张量
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  // 前向传播：对每个输入张量计算线性变换
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 工厂方法：从 RuntimeOperator 创建 LinearLayer
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& linear_layer);

  // 从一维浮点向量设置权重
  void set_weights(const std::vector<float>& weights) override;

  // 从张量向量设置权重（委托给基类）
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

 private:
  int32_t in_features_ = 0;   // 输入特征数
  int32_t out_features_ = 0;  // 输出特征数
  bool use_bias_ = false;     // 是否使用偏置
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_LINEAR_HPP_
