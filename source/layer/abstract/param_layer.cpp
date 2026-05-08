#include "layer/abstract/param_layer.hpp"
#include <glog/logging.h>

namespace net_infer {

/**
 * @brief 构造函数，初始化参数层
 * @param layer_name 层名称
 */
ParamLayer::ParamLayer(const std::string& layer_name) : Layer(layer_name) {}

/**
 * @brief 初始化偏置参数张量
 * @param param_count 偏置张量的数量（例如卷积核的数量）
 * @param param_channel 每个偏置张量的通道数
 * @param param_height 每个偏置张量的高度
 * @param param_width 每个偏置张量的宽度
 *
 * 内部会为每个偏置分配一个指定形状的 ftensor，并存储到 bias_ 列表中。
 */
void ParamLayer::InitBiasParam(const uint32_t param_count, const uint32_t param_channel,
                               const uint32_t param_height, const uint32_t param_width) {
  this->bias_ = std::vector<sftensor>(param_count);
  for (uint32_t i = 0; i < param_count; ++i) {
    this->bias_.at(i) = std::make_shared<ftensor>(param_channel, param_height, param_width);
  }
}

/**
 * @brief 初始化权重参数张量
 * @param param_count 权重张量的数量（例如卷积核的数量）
 * @param param_channel 每个权重张量的通道数
 * @param param_height 每个权重张量的高度
 * @param param_width 每个权重张量的宽度
 *
 * 内部会为每个权重分配一个指定形状的 ftensor，并存储到 weights_ 列表中。
 */
void ParamLayer::InitWeightParam(const uint32_t param_count, const uint32_t param_channel,
                                 const uint32_t param_height, const uint32_t param_width) {
  this->weights_ = std::vector<sftensor>(param_count);
  for (uint32_t i = 0; i < param_count; ++i) {
    this->weights_.at(i) = std::make_shared<ftensor>(param_channel, param_height, param_width);
  }
}

/**
 * @brief 获取权重张量列表（常量引用）
 * @return 权重张量列表
 */
const std::vector<std::shared_ptr<Tensor<float>>>& ParamLayer::weights() const {
  return this->weights_;
}

/**
 * @brief 获取偏置张量列表（常量引用）
 * @return 偏置张量列表
 */
const std::vector<std::shared_ptr<Tensor<float>>>& ParamLayer::bias() const { return this->bias_; }

/**
 * @brief 使用张量列表设置权重
 * @param weights 待设置的权重张量列表
 *
 * 会对每个权重的形状进行校验，确保与当前层的权重形状一致后再赋值。
 */
void ParamLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  CHECK(weights.size() == weights_.size());
  for (uint32_t i = 0; i < weights.size(); ++i) {
    CHECK(this->weights_.at(i) != nullptr);
    CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
    CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
    CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
  }
  this->weights_ = weights;
}

/**
 * @brief 使用张量列表设置偏置
 * @param bias 待设置的偏置张量列表
 *
 * 若当前层已有偏置，会对每个偏置的形状进行校验，确保一致后再赋值。
 */
void ParamLayer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
  if (!this->bias_.empty()) {
    CHECK(bias.size() == bias_.size());
    for (uint32_t i = 0; i < bias.size(); ++i) {
      CHECK(this->bias_.at(i) != nullptr);
      CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
      CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
      CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
    }
    this->bias_ = bias;
  }
}

/**
 * @brief 使用一维浮点向量设置权重
 * @param weights 扁平化的一维权重数据
 *
 * 要求 weights 的总元素数等于当前所有权重的元素总数，
 * 并均匀切分到每个权重张量中。
 */
void ParamLayer::set_weights(const std::vector<float>& weights) {
  size_t weight_size = 0;
  const size_t elem_size = weights.size();

  // 统计当前所有权重的总元素数
  const uint32_t batch_size = this->weights_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    weight_size += this->weights_.at(i)->size();
  }

  CHECK_EQ(weight_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);

  // 将一维数据均匀切分并填充到各个权重张量
  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values =
        std::vector<float>{weights.begin() + start_offset, weights.begin() + end_offset};
    this->weights_.at(idx)->Fill(sub_values);
  }
}

/**
 * @brief 按索引获取单个权重张量
 * @param index 权重索引
 * @return 对应索引的权重张量
 */
std::shared_ptr<Tensor<float>> ParamLayer::weight(int32_t index) const {
  CHECK_LE(index, this->weights_.size());
  return this->weights_.at(index);
}

/**
 * @brief 使用一维浮点向量设置偏置
 * @param bias 扁平化的一维偏置数据
 *
 * 要求 bias 的总元素数等于当前所有偏置的元素总数，
 * 并均匀切分到每个偏置张量中。
 */
void ParamLayer::set_bias(const std::vector<float>& bias) {
  size_t bias_size = 0;
  const size_t elem_size = bias.size();

  // 统计当前所有偏置的总元素数
  const uint32_t batch_size = this->bias_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    bias_size += this->bias_.at(i)->size();
  }

  CHECK_EQ(bias_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);

  // 将一维数据均匀切分并填充到各个偏置张量
  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values =
        std::vector<float>{bias.begin() + start_offset, bias.begin() + end_offset};
    this->bias_.at(idx)->Fill(sub_values);
  }
}

}  // namespace net_infer
