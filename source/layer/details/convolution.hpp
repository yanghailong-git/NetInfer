#ifndef NET_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#define NET_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#include "base_convolution.hpp"
#include "layer/abstract/param_layer.hpp"

namespace net_infer {

/// 标准二维卷积层，采用 im2col + GEMM 方法实现。
class ConvolutionLayer : public BaseConvolutionLayer {
 public:
  explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h,
                            uint32_t kernel_w, uint32_t padding_h, uint32_t padding_w,
                            uint32_t stride_h, uint32_t stride_w, uint32_t groups,
                            bool use_bias = true, uint32_t output_padding_h = 0,
                            uint32_t output_padding_w = 0, uint32_t dilation_h = 1,
                            uint32_t dilation_w = 1)
      : BaseConvolutionLayer(ConvType::kOpConv, output_channel, in_channel, kernel_h, kernel_w,
                             padding_h, padding_w, stride_h, stride_w, groups, use_bias,
                             output_padding_h, output_padding_w, dilation_h, dilation_w) {}

 private:
  /// 若卷积核为 1x1、无填充、步长为 1 且空洞率为 1，则返回 true。
  bool Is1x1KernelNoPadding(uint32_t kernel_h, uint32_t kernel_w) const;

  /// 将每个卷积核展平为行/列向量，为基于 GEMM 的卷积做准备。
  void InitIm2ColWeight() override;

  /// 通过 im2col + GEMM 计算单个分组的卷积输出。
  void ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h, uint32_t kernel_w,
                     uint32_t kernel_count_group, uint32_t input_h, uint32_t input_w,
                     uint32_t channels_per_group, uint32_t output_h, uint32_t output_w,
                     uint32_t group) const override;

  /// 根据输入尺寸、卷积核尺寸、填充、步长和空洞率计算空间输出尺寸。
  std::pair<uint32_t, uint32_t> ComputeOutputSize(uint32_t input_h, uint32_t input_w,
                                                  uint32_t kernel_h,
                                                  uint32_t kernel_w) const override;

  /// 执行 im2col 输入矩阵与卷积核矩阵的 GEMM 乘法，然后加上偏置。
  void ConvGEMMBias(const arma::fmat& input_matrix, sftensor output_tensor, uint32_t group,
                    uint32_t kernel_index, uint32_t kernel_count_group, uint32_t output_h,
                    uint32_t output_w, bool is_1x1conv_nopadding) const;

  /// 将输入张量的局部区域转换为适合 GEMM 的列优先矩阵。
  [[nodiscard]] arma::fmat ConvIm2Col(sftensor input, uint32_t kernel_h, uint32_t kernel_w,
                                      uint32_t input_h, uint32_t input_w,
                                      uint32_t channels_per_group, uint32_t output_h,
                                      uint32_t output_w, uint32_t group, uint32_t row_len,
                                      uint32_t col_len) const;
};

}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
