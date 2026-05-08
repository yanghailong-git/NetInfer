#include "deconvolution.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace net_infer {

// 暂不支持通过Tensor指针数组设置反卷积权重
void DeconvolutionLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << "The set weights function does not support this convolution type: "
             << int32_t(conv_type_);
}

// 从一维浮点数组中设置反卷积层的权重
// 将原始权重数据按group分组，并重新排列为c n h w -> n c h w的存储格式
void DeconvolutionLayer::set_weights(const std::vector<float>& weights) {
  const uint32_t kernel_count = this->weights_.size();

  CHECK_GT(kernel_count, 0);
  // 每个group内的卷积核数量
  const uint32_t kernel_count_group = kernel_count / groups_;
  // 每个卷积核的输入通道数
  const uint32_t kernel_channel = this->weights_.at(0)->channels();
  uint32_t kernel_height = this->weights_.at(0)->rows();
  uint32_t kernel_width = this->weights_.at(0)->cols();
  // 根据空洞率(dilation)计算实际卷积核尺寸
  if (dilation_h_ > 1) {
    kernel_height = (kernel_height + dilation_h_ - 1) / dilation_h_;
  }
  if (dilation_w_ > 1) {
    kernel_width = (kernel_width + dilation_w_ - 1) / dilation_w_;
  }

  // 校验权重总数量是否正确
  CHECK_EQ(kernel_count * kernel_channel * kernel_width * kernel_height, weights.size());

  const uint32_t kernel_hw = kernel_height * kernel_width;
  const uint32_t kernel_nhw = kernel_count_group * kernel_hw;
  const uint32_t kernel_plane = kernel_channel * kernel_nhw;

  for (uint32_t group = 0; group < groups_; ++group) {
    // sub_weights表示一个group内所有卷积核的权重值
    std::vector<float> sub_weights(kernel_plane);
    std::copy(weights.data() + group * kernel_plane, weights.data() + (group + 1) * kernel_plane,
              sub_weights.begin());
    for (uint32_t kg = 0; kg < kernel_count_group; ++kg) {
      const uint32_t channel_offset = kg * kernel_hw;
      const uint32_t kernel_idx = group * kernel_count_group + kg;
      /*
       * 卷积核权重摆放的顺序是c n h w， 需要将它调整到n c h w
       * 其中n表示卷积核次序，kernel_idx = group * kernel_count_group + kg;
       * origin_pixel_idx = ic * kernel_nhw (nhw) + kg(n) * kernel_hw + ...
       */
      for (uint32_t ic = 0; ic < kernel_channel; ++ic) {
        const uint32_t kernel_offset = ic * kernel_nhw;
        arma::fmat& kernel_channel_mat = this->weights_.at(kernel_idx)->slice(ic);

        for (uint32_t kw = 0; kw < kernel_width; ++kw) {
          uint32_t kw_dilation = kw * dilation_w_;
          float* kernel_ptr = kernel_channel_mat.colptr(kw_dilation);
          for (uint32_t kh = 0; kh < kernel_height; ++kh) {
            uint32_t kh_dilation = kh * dilation_h_;
            *(kernel_ptr + kh_dilation) =
                sub_weights.at(kernel_offset + channel_offset + kh * kernel_width + kw);
          }
        }
      }
    }
  }
}

// 计算单个group的反卷积输出
// 对每个输出通道分别进行GEMM计算和Col2Im转换
void DeconvolutionLayer::ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h,
                                       uint32_t kernel_w, uint32_t kernel_count_group,
                                       uint32_t input_h, uint32_t input_w,
                                       uint32_t channels_per_group, uint32_t output_h,
                                       uint32_t output_w, uint32_t group) const {
#pragma omp parallel for
  for (uint32_t k = 0; k < kernel_count_group; ++k) {
    const arma::fmat& gemm_result =
        DeconvGEMM(input, input_h, input_w, channels_per_group, group, k, kernel_count_group);
    DeconvCol2ImBias(gemm_result, output_tensor, input_h, input_w, group, k, kernel_count_group,
                     kernel_h, kernel_w, output_h, output_w);
  }
}

// 根据输入尺寸、卷积核尺寸、步长、填充和输出填充计算反卷积后的输出尺寸
std::pair<uint32_t, uint32_t> DeconvolutionLayer::ComputeOutputSize(const uint32_t input_h,
                                                                    const uint32_t input_w,
                                                                    const uint32_t kernel_h,
                                                                    const uint32_t kernel_w) const {
  uint32_t output_h = 0;
  uint32_t output_w = 0;

  output_h = (input_h - 1) * stride_h_ + kernel_h + output_padding_h_;
  output_w = (input_w - 1) * stride_w_ + kernel_w + output_padding_w_;
  CHECK(output_h > 2 * padding_h_ && output_w > 2 * padding_w_);
  output_h -= 2 * padding_h_;
  output_w -= 2 * padding_w_;
  return {output_h, output_w};
}

// 执行反卷积的GEMM（通用矩阵乘法）操作
// 将输入特征图和卷积核权重进行矩阵乘法，得到中间结果
arma::fmat DeconvolutionLayer::DeconvGEMM(const sftensor& input, uint32_t input_h, uint32_t input_w,
                                          uint32_t channels_per_group, uint32_t group,
                                          uint32_t kernel_index,
                                          uint32_t kernel_count_group) const {
  CHECK(input != nullptr && !input->empty());

  kernel_index = kernel_index + group * kernel_count_group;
  CHECK(kernel_index < this->weights_.size());

  sftensor group_kernel = this->weights_.at(kernel_index);
  CHECK(group_kernel != nullptr && !group_kernel->empty());

  uint32_t input_hw = input_h * input_w;
  uint32_t kernel_hw = group_kernel->rows() * group_kernel->cols();

  // 将输入数据按group和channel组织为矩阵（不复制数据，直接引用）
  arma::fmat multi_input_channel(input->matrix_raw_ptr(group * channels_per_group), input_hw,
                                 channels_per_group, false, true);

  // 将卷积核权重组织为矩阵（不复制数据，直接引用）
  arma::fmat multi_kernel_channel(group_kernel->raw_ptr(), kernel_hw, channels_per_group, false,
                                  true);
  return multi_kernel_channel * (multi_input_channel.t());
}

// 将GEMM结果通过Col2Im方式映射到输出特征图，并加上偏置
void DeconvolutionLayer::DeconvCol2ImBias(const arma::fmat& gemm_result, sftensor output_tensor,
                                          uint32_t input_h, uint32_t input_w, uint32_t group,
                                          uint32_t kernel_index, uint32_t kernel_count_group,
                                          uint32_t kernel_h, uint32_t kernel_w, uint32_t output_h,
                                          uint32_t output_w) const {
  CHECK(!gemm_result.empty());
  CHECK(input_h > 0 && input_w > 0);
  CHECK(output_tensor != nullptr && !output_tensor->empty());

  // 创建带padding的输出缓冲区
  arma::fmat output_padding(output_h + 2 * padding_h_, output_w + 2 * padding_w_);

  uint32_t slide_w = input_w;
  uint32_t slide_h = input_h;
  uint32_t slide_size = slide_h * slide_w;
  // 遍历输入特征图的每个位置，将对应的GEMM列向量按卷积核尺寸展开并累加到输出位置
  for (uint32_t index = 0; index < slide_size; ++index) {
    uint32_t x = index / slide_h;
    uint32_t y = index % slide_h;
    const uint32_t offset_x = x * stride_w_;
    const uint32_t offset_y = y * stride_h_;
    // 将GEMM结果的一列重塑为kernel_h x kernel_w的矩阵
    arma::fmat gemm_column(const_cast<float*>(gemm_result.colptr(index)), kernel_h, kernel_w, false,
                           true);

    uint32_t gemm_rows = gemm_column.n_rows;
    uint32_t gemm_cols = gemm_column.n_cols;

    // 将展开的卷积核数据累加到对应输出位置
    for (uint32_t col = 0; col < gemm_cols; ++col) {
      float* column_ptr = gemm_column.colptr(col);
      float* output_ptr = output_padding.colptr(offset_x + col);
      for (uint32_t row = 0; row < gemm_rows; ++row) {
        *(output_ptr + offset_y + row) += *(column_ptr + row);
      }
    }
  }

  // 去掉padding，得到最终输出区域
  kernel_index = kernel_index + group * kernel_count_group;
  arma::fmat output(output_tensor->matrix_raw_ptr(kernel_index), output_h, output_w, false, true);

  output = output_padding.submat(padding_h_, padding_w_, output_h + padding_h_ - 1,
                                 output_w + padding_w_ - 1);

  // 加上对应通道的偏置
  return AddBias(output, kernel_index);
}

// 注册反卷积层（对应PyTorch的nn.ConvTranspose2d）
LayerRegistererWrapper kDeConvCreateInstance(BaseConvolutionLayer::CreateInstance,
                                             "nn.ConvTranspose2d");
}  // namespace net_infer
