#ifndef NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"
namespace net_infer {

/**
 * @brief 均方根（RMS）归一化层。
 *
 * 通过将输入除以其均方根进行归一化，然后
 * 应用可学习的缩放（增益）参数。
 * 公式：output = gain * (input / sqrt(mean(input^2) + eps))。
 */
class RMSNormLayer : public ParamLayer {
 public:
  explicit RMSNormLayer();

  /**
   * @brief 前向传播：对每个输入张量应用 RMS 归一化。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次（若为空则会创建）。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief 从共享张量指针向量设置层权重。
   * @param weights 权重张量（期望恰好一个增益张量）。
   */
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  /**
   * @brief 从一维浮点向量设置层权重（不支持）。
   * @param weights 扁平权重数据。
   */
  void set_weights(const std::vector<float>& weights) override;
 private:
  float eps_ = 1e-5f;  ///< 用于数值稳定性的小常数。
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
