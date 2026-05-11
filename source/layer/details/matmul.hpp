#ifndef NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
#include "layer/abstract/param_layer.hpp"
namespace net_infer {

/**
 * @brief 用于 LLaMA 风格模型的矩阵乘法层。
 *
 * 该层在输入张量与存储的权重张量之间执行矩阵乘法（w @ x）。
 * 支持一维和二维输入形状。
 */
class LLamaMatmulLayer : public ParamLayer {
 public:
  /**
   * @brief 构造函数。
   * @param weight_dim0 权重矩阵的行数（输出特征数）。
   * @param weight_dim1 权重矩阵的列数（输入特征数）。
   */
  explicit LLamaMatmulLayer(int32_t weight_dim0, int32_t weight_dim1);

  /**
   * @brief 前向传播：对每个输入张量计算矩阵乘法。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次（若为空则会调整大小/创建）。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief 从共享张量指针向量设置层权重。
   * @param weights 权重张量（期望恰好一个）。
   */
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  /**
   * @brief 从一维浮点向量设置层权重（不支持）。
   * @param weights 扁平权重数据。
   */
  void set_weights(const std::vector<float>& weights) override;

 private:
  int32_t weight_dim0_ = 0;  ///< 输出维度（权重矩阵的行数）。
  int32_t weight_dim1_ = 0;  ///< 输入维度（权重矩阵的列数）。
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
