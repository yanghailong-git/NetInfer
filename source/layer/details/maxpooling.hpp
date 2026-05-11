#ifndef NET_INFER_SOURCE_LAYER_MAX_POOLING_
#define NET_INFER_SOURCE_LAYER_MAX_POOLING_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief 二维最大池化层。
 *
 * 从输入张量空间维度（高/宽）上滑动的每个池化窗口中提取最大值。
 */
class MaxPoolingLayer : public NonParamLayer {
 public:
  /**
   * @brief 构造函数。
   * @param padding_h 高度方向的填充。
   * @param padding_w 宽度方向的填充。
   * @param pooling_size_h 池化窗口的高度。
   * @param pooling_size_w 池化窗口的宽度。
   * @param stride_h 池化窗口的垂直步长。
   * @param stride_w 池化窗口的水平步长。
   */
  explicit MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                           uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w);

  /**
   * @brief 前向传播：对每个输入张量应用最大池化。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次（若为空则会创建）。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief 在前向传播前校验输入和输出张量数组。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /**
   * @brief 工厂方法，根据运行时算子参数创建 MaxPoolingLayer。
   * @param op 包含 "stride"、"padding" 和 "kernel_size" 参数的运行时算子。
   * @param max_layer 待赋值的输出层指针。
   * @return 表示成功或失败的状态码。
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& max_layer);

 private:
  uint32_t padding_h_ = 0;      ///< 高度方向填充。
  uint32_t padding_w_ = 0;      ///< 宽度方向填充。
  uint32_t pooling_size_h_ = 0; ///< 池化核的高度。
  uint32_t pooling_size_w_ = 0; ///< 池化核的宽度。
  uint32_t stride_h_ = 1;       ///< 垂直步长。
  uint32_t stride_w_ = 1;       ///< 水平步长。
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_MAX_POOLING_
