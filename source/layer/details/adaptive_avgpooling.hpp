#ifndef NET_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
#define NET_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/// 自适应平均池化层，将每个输入特征图降采样到固定的输出尺寸 (output_h, output_w)。
/// 池化区域大小和步长根据输入尺寸自动计算。
class AdaptiveAveragePoolingLayer : public NonParamLayer {
 public:
  explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

  /// 校验输入/输出张量数组，并检查输出尺寸是否设置正确。
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /// 对批次中的每个张量执行自适应平均池化。
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /// 工厂方法，根据运行时算子定义创建实例。
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& avg_layer);

 private:
  uint32_t output_h_ = 0;  ///< 目标输出高度。
  uint32_t output_w_ = 0;  ///< 目标输出宽度。
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
