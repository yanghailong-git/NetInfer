#ifndef NET_INFER_SOURCE_LAYER_DETAILS_DECONVOLUTION_H
#define NET_INFER_SOURCE_LAYER_DETAILS_DECONVOLUTION_H
#include "base_convolution.hpp"
#include "data/tensor.hpp"
namespace net_infer {

// 反卷积层（转置卷积层）
// 继承自BaseConvolutionLayer，实现了二维反卷积操作
// 对应PyTorch中的nn.ConvTranspose2d
class DeconvolutionLayer : public BaseConvolutionLayer {
 public:
  // 构造函数
  // output_channel: 输出通道数
  // in_channel: 输入通道数
  // kernel_h, kernel_w: 卷积核高和宽
  // padding_h, padding_w: 填充大小
  // stride_h, stride_w: 步长
  // groups: 分组数
  // use_bias: 是否使用偏置
  // output_padding_h, output_padding_w: 输出填充大小
  // dilation_h, dilation_w: 空洞率
  explicit DeconvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h,
                              uint32_t kernel_w, uint32_t padding_h, uint32_t padding_w,
                              uint32_t stride_h, uint32_t stride_w, uint32_t groups,
                              bool use_bias = true, uint32_t output_padding_h = 0,
                              uint32_t output_padding_w = 0, uint32_t dilation_h = 1,
                              uint32_t dilation_w = 1)
      : BaseConvolutionLayer(ConvType::kOpDeconv, output_channel, in_channel, kernel_h, kernel_w,
                             padding_h, padding_w, stride_h, stride_w, groups, use_bias,
                             output_padding_h, output_padding_w, dilation_h, dilation_w) {}

  // 从一维浮点数组设置权重
  void set_weights(const std::vector<float>& weights) override;

  // 从Tensor指针数组设置权重（当前不支持）
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

 private:
  // 计算单个group的反卷积输出
  void ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h, uint32_t kernel_w,
                     uint32_t kernel_count_group, uint32_t input_h, uint32_t input_w,
                     uint32_t channels_per_group, uint32_t output_h, uint32_t output_w,
                     uint32_t group) const override;

  // 根据输入尺寸和卷积核尺寸计算输出特征图尺寸
  std::pair<uint32_t, uint32_t> ComputeOutputSize(uint32_t input_h, uint32_t input_w,
                                                  uint32_t kernel_h,
                                                  uint32_t kernel_w) const override;

  // 将GEMM结果通过Col2Im映射到输出特征图并加上偏置
  void DeconvCol2ImBias(const arma::fmat& gemm_result, sftensor output_tensor, uint32_t input_h,
                        uint32_t input_w, uint32_t group, uint32_t kernel_index,
                        uint32_t kernel_count_group, uint32_t kernel_h, uint32_t kernel_w,
                        uint32_t output_h, uint32_t output_w) const;

  // 执行反卷积的GEMM操作：卷积核矩阵 × 输入矩阵转置
  [[nodiscard]] arma::fmat DeconvGEMM(const sftensor& input, uint32_t input_h, uint32_t input_w,
                                      uint32_t channels_per_group, uint32_t group,
                                      uint32_t kernel_index, uint32_t kernel_count_group) const;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_DECONVOLUTION_H
