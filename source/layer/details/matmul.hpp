#ifndef NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
#include "layer/abstract/param_layer.hpp"
namespace net_infer {
class LLamaMatmulLayer : public ParamLayer {
 public:
  explicit LLamaMatmulLayer(int32_t weight_dim0, int32_t weight_dim1);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  void set_weights(const std::vector<float>& weights) override;

 private:
  int32_t weight_dim0_ = 0;
  int32_t weight_dim1_ = 0;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
