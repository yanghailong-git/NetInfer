#ifndef NET_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/// 拼接层，沿指定维度将多个输入张量合并。
class CatLayer : public NonParamLayer {
 public:
  explicit CatLayer(int32_t dim);

  /// 沿配置的维度拼接输入张量，并将结果存入 outputs。
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /// 校验输入是否有效，以及拼接维度是否受支持。
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /// 工厂方法，根据运行时算子定义创建 CatLayer。
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& cat_layer);

 private:
  int32_t dim_ = 0;  ///< 拼接维度（1 或 -3 表示通道轴）。
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
