#ifndef NET_INFER_SOURCE_LAYER_DETAILS_BASE_CONVOLUTION_H
#define NET_INFER_SOURCE_LAYER_DETAILS_BASE_CONVOLUTION_H
#include "layer/abstract/param_layer.hpp"
namespace net_infer {

/// 卷积操作类型枚举
enum class ConvType {
  kOpConvUnknown = -1,
  kOpConv = 0,    // 普通卷积
  kOpDeconv = 1,  // 转置卷积
};

/// 卷积层的基类，封装了卷积/转置卷积的公共参数和逻辑
class BaseConvolutionLayer : public ParamLayer {
 public:
  explicit BaseConvolutionLayer(ConvType conv_type, uint32_t output_channel, uint32_t in_channel,
                                uint32_t kernel_h, uint32_t kernel_w, uint32_t padding_h,
                                uint32_t padding_w, uint32_t stride_h, uint32_t stride_w,
                                uint32_t groups, bool use_bias = true,
                                uint32_t output_padding_h = 0, uint32_t output_padding_w = 0,
                                uint32_t dilation_h = 1, uint32_t dilation_w = 1);

  /// 从运行时算子中解析参数并创建卷积层实例
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& conv_layer);

  /// 前向推理：对输入批量中的每个张量执行卷积运算
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

 private:
  /// 纯虚函数：计算单组的卷积输出，由子类（普通卷积/转置卷积）实现具体逻辑
  virtual void ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h,
                             uint32_t kernel_w, uint32_t kernel_count_group, uint32_t input_h,
                             uint32_t input_w, uint32_t input_c_group, uint32_t output_h,
                             uint32_t output_w, uint32_t group) const = 0;

  /// 纯虚函数：根据输入尺寸和卷积核尺寸计算输出尺寸
  virtual std::pair<uint32_t, uint32_t> ComputeOutputSize(uint32_t input_h, uint32_t input_w,
                                                          uint32_t kernel_h,
                                                          uint32_t kernel_w) const = 0;

 public:
  /// 校验输入输出张量、权重及参数是否合法
  StatusCode Check(const std::vector<sftensor>& inputs, const std::vector<sftensor>& outputs);

 private:
  /// 初始化 im2col 所需的权重矩阵，默认空实现，子类可覆盖
  virtual void InitIm2ColWeight();

 protected:
  /// 为输出矩阵加上指定索引的偏置值
  void AddBias(arma::fmat& output, uint32_t bias_index) const;

 protected:
  uint32_t groups_ = 1;          // 分组卷积的组数
  bool use_bias_ = false;        // 是否使用偏置

  uint32_t padding_h_ = 0;       // 高度方向的填充
  uint32_t padding_w_ = 0;       // 宽度方向的填充
  uint32_t stride_h_ = 1;        // 高度方向的步长
  uint32_t stride_w_ = 1;        // 宽度方向的步长

  uint32_t output_padding_h_ = 0;  // 转置卷积输出填充（高度）
  uint32_t output_padding_w_ = 0;  // 转置卷积输出填充（宽度）

  uint32_t dilation_h_ = 1;      // 高度方向空洞率
  uint32_t dilation_w_ = 1;      // 宽度方向空洞率

  ConvType conv_type_ = ConvType::kOpConvUnknown;  // 卷积类型
  std::vector<arma::fmat> kernel_matrix_arr_;      // 用于 im2col 的权重矩阵缓存
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_BASE_CONVOLUTION_H
