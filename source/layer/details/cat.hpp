#ifndef NET_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/// Concatenation layer that merges multiple input tensors along a specified dimension.
class CatLayer : public NonParamLayer {
 public:
  explicit CatLayer(int32_t dim);

  /// Concatenates input tensors along the configured dimension and stores results in outputs.
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /// Validates that inputs are valid and that the concat dimension is supported.
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /// Factory method to create a CatLayer from a runtime operator definition.
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& cat_layer);

 private:
  int32_t dim_ = 0;  ///< Dimension along which to concatenate (1 or -3 for channel axis).
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
