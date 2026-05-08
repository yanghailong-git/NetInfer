#ifndef NET_INFER_SOURCE_LAYER_FLATTEN_HPP_
#define NET_INFER_SOURCE_LAYER_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace net_infer {

/**
 * @brief 视图层（View Layer）
 *
 * 通过改变张量的形状（reshape）来实现数据的重新组织，不修改底层数据。
 * 支持在 shape 中使用 -1 表示自动推断该维度大小（仅允许出现一次，且建议位于末尾）。
 */
class ViewLayer : public NonParamLayer {
 public:
  explicit ViewLayer(std::vector<int32_t> shapes);

  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& view_layer);

 private:
  std::vector<int32_t> shapes_;  // 目标形状，-1 表示自动推断
};

}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_FLATTEN_HPP_
