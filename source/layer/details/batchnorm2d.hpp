#ifndef NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
#define NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_

#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"

namespace net_infer {

/// 二维特征图的批归一化层（NCHW 格式）。
/// 使用运行均值和方差对每个通道进行归一化，随后执行仿射变换。
class BatchNorm2dLayer : public ParamLayer {
 public:
  explicit BatchNorm2dLayer(uint32_t num_features, float eps, std::vector<float> affine_weight,
                            std::vector<float> affine_bias);

  /// 前向传播：对输入的每个通道进行归一化，并应用学习到的仿射缩放和平移。
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /// 工厂方法，从运行时算子构造 BatchNorm2dLayer。
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& batch_layer);

 private:
  float eps_ = 1e-5f;              ///< 加到方差上的小常数，用于数值稳定性。
  std::vector<float> affine_weight_;  ///< 每个通道的 gamma（缩放）参数。
  std::vector<float> affine_bias_;    ///< 每个通道的 beta（平移）参数。
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
