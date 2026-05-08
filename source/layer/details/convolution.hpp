#ifndef NET_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#define NET_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#include "base_convolution.hpp"
#include "layer/abstract/param_layer.hpp"

namespace net_infer {

/// Standard 2D convolution layer using the im2col + GEMM approach.
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
  /// Returns true if the convolution is a 1x1 kernel with no padding, stride 1, and dilation 1.
  bool Is1x1KernelNoPadding(uint32_t kernel_h, uint32_t kernel_w) const;

  /// Flattens each kernel into a row/column vector to prepare for GEMM-based convolution.
  void InitIm2ColWeight() override;

  /// Computes the convolution output for a single group by im2col + GEMM.
  void ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h, uint32_t kernel_w,
                     uint32_t kernel_count_group, uint32_t input_h, uint32_t input_w,
                     uint32_t channels_per_group, uint32_t output_h, uint32_t output_w,
                     uint32_t group) const override;

  /// Calculates the spatial output size from input size, kernel size, padding, stride, and dilation.
  std::pair<uint32_t, uint32_t> ComputeOutputSize(uint32_t input_h, uint32_t input_w,
                                                  uint32_t kernel_h,
                                                  uint32_t kernel_w) const override;

  /// Performs GEMM multiplication between the im2col input matrix and the kernel matrix, then adds bias.
  void ConvGEMMBias(const arma::fmat& input_matrix, sftensor output_tensor, uint32_t group,
                    uint32_t kernel_index, uint32_t kernel_count_group, uint32_t output_h,
                    uint32_t output_w, bool is_1x1conv_nopadding) const;

  /// Converts a local region of the input tensor into a column-major matrix suitable for GEMM.
  [[nodiscard]] arma::fmat ConvIm2Col(sftensor input, uint32_t kernel_h, uint32_t kernel_w,
                                      uint32_t input_h, uint32_t input_w,
                                      uint32_t channels_per_group, uint32_t output_h,
                                      uint32_t output_w, uint32_t group, uint32_t row_len,
                                      uint32_t col_len) const;
};

}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
