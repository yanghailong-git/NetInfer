#include "data/tensor.hpp"

namespace net_infer {

// ------------------- 构造函数 -------------------

/**
 * @brief 使用外部原始指针构造一维张量（共享内存，不拷贝数据）
 * @param raw_ptr 外部原始数据指针
 * @param size 元素个数
 *
 * 内部使用 arma::Cube 包装指针，形状视为 {size}。
 */
template <typename T>
Tensor<T>::Tensor(T* raw_ptr, uint32_t size) {
  CHECK_NE(raw_ptr, nullptr);
  this->raw_shapes_ = {size};
  this->data_ = arma::Cube<T>(raw_ptr, 1, size, 1, false, true);
}

/**
 * @brief 使用外部原始指针构造二维张量（共享内存，不拷贝数据）
 * @param raw_ptr 外部原始数据指针
 * @param rows 行数
 * @param cols 列数
 *
 * 当 rows == 1 时，raw_shapes_ 记为一维 {cols}，否则为二维 {rows, cols}。
 */
template <typename T>
Tensor<T>::Tensor(T* raw_ptr, uint32_t rows, uint32_t cols) {
  CHECK_NE(raw_ptr, nullptr);
  this->data_ = arma::Cube<T>(raw_ptr, rows, cols, 1, false, true);
  if (rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  }
}

/**
 * @brief 使用外部原始指针构造三维张量（共享内存，不拷贝数据）
 * @param raw_ptr 外部原始数据指针
 * @param channels 通道数
 * @param rows 行数
 * @param cols 列数
 *
 * 根据维度是否为 1 来简化 raw_shapes_ 的表示。
 */
template <typename T>
Tensor<T>::Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols) {
  CHECK_NE(raw_ptr, nullptr);
  this->data_ = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

/**
 * @brief 使用外部原始指针按 shape 向量构造张量（共享内存，不拷贝数据）
 * @param raw_ptr 外部原始数据指针
 * @param shapes 形状向量，大小必须为 3，依次表示 [channels, rows, cols]
 *
 * 内部根据维度是否为 1 来简化 raw_shapes_ 的表示。
 */
template <typename T>
Tensor<T>::Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes) {
  CHECK_EQ(shapes.size(), 3);
  uint32_t channels = shapes.at(0);
  uint32_t rows = shapes.at(1);
  uint32_t cols = shapes.at(2);

  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }

  this->data_ = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
}

/**
 * @brief 构造一个指定三维形状的新张量（内部分配内存）
 * @param channels 通道数
 * @param rows 行数
 * @param cols 列数
 */
template <typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::Cube<T>(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

/**
 * @brief 构造一个一维张量（内部分配内存）
 * @param size 元素个数
 */
template <typename T>
Tensor<T>::Tensor(uint32_t size) {
  data_ = arma::Cube<T>(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

/**
 * @brief 构造一个二维张量（内部分配内存）
 * @param rows 行数
 * @param cols 列数
 */
template <typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::Cube<T>(rows, cols, 1);
  if (rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  }
}

/**
 * @brief 使用 shape 向量构造张量（内部分配内存）
 * @param shapes 形状向量，长度范围为 [1, 3]
 *
 * 内部会将不足 3 维的 shape 左侧补 1，再映射到 arma::Cube 的 [channels, rows, cols]。
 */
template <typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);

  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  data_ = arma::Cube<T>(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

// ------------------- 形状与尺寸查询 -------------------

/**
 * @brief 返回张量的行数
 * @return 行数
 */
template <typename T>
uint32_t Tensor<T>::rows() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->data_.n_rows;
}

/**
 * @brief 返回张量的列数
 * @return 列数
 */
template <typename T>
uint32_t Tensor<T>::cols() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->data_.n_cols;
}

/**
 * @brief 返回张量的通道数
 * @return 通道数
 */
template <typename T>
uint32_t Tensor<T>::channels() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->data_.n_slices;
}

/**
 * @brief 返回张量中的元素总数
 * @return 元素总数
 */
template <typename T>
size_t Tensor<T>::size() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->data_.size();
}

/**
 * @brief 返回单个通道平面的大小（rows * cols）
 * @return 单个通道的元素个数
 */
template <typename T>
size_t Tensor<T>::plane_size() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->rows() * this->cols();
}

// ------------------- 数据访问与设置 -------------------

/**
 * @brief 使用 arma::Cube 设置张量的底层数据
 * @param data 待设置的 arma::Cube 数据
 *
 * 要求 data 的维度必须与当前张量完全一致。
 */
template <typename T>
void Tensor<T>::set_data(const arma::Cube<T>& data) {
  CHECK(data.n_rows == this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

/**
 * @brief 判断张量是否为空（未分配数据）
 * @return true 表示为空，false 表示已分配数据
 */
template <typename T>
bool Tensor<T>::empty() const {
  return this->data_.empty();
}

/**
 * @brief 按线性偏移常量访问张量元素
 * @param offset 线性偏移量
 * @return 对应位置的元素值
 */
template <typename T>
const T Tensor<T>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

/**
 * @brief 按线性偏移访问张量元素（可修改）
 * @param offset 线性偏移量
 * @return 对应位置元素的引用
 */
template <typename T>
T& Tensor<T>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

/**
 * @brief 获取张量的完整形状 [channels, rows, cols]
 * @return 包含三个维度的向量
 */
template <typename T>
std::vector<uint32_t> Tensor<T>::shapes() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return {this->channels(), this->rows(), this->cols()};
}

/**
 * @brief 获取底层 arma::Cube 数据的引用（可修改）
 * @return arma::Cube<T> 的引用
 */
template <typename T>
arma::Cube<T>& Tensor<T>::data() {
  return this->data_;
}

/**
 * @brief 获取底层 arma::Cube 数据的常量引用
 * @return arma::Cube<T> 的常量引用
 */
template <typename T>
const arma::Cube<T>& Tensor<T>::data() const {
  return this->data_;
}

/**
 * @brief 获取指定通道的二维矩阵切片（可修改）
 * @param channel 通道索引
 * @return 该通道对应的 arma::Mat<T> 引用
 */
template <typename T>
arma::Mat<T>& Tensor<T>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

/**
 * @brief 获取指定通道的二维矩阵切片（只读）
 * @param channel 通道索引
 * @return 该通道对应的 arma::Mat<T> 常量引用
 */
template <typename T>
const arma::Mat<T>& Tensor<T>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

/**
 * @brief 按三维索引访问元素（只读）
 * @param channel 通道索引
 * @param row 行索引
 * @param col 列索引
 * @return 对应位置的元素值
 */
template <typename T>
const T Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

/**
 * @brief 按三维索引访问元素（可修改）
 * @param channel 通道索引
 * @param row 行索引
 * @param col 列索引
 * @return 对应位置元素的引用
 */
template <typename T>
T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

// ------------------- 张量变换操作 -------------------

/**
 * @brief 对张量进行边界填充（Padding）
 * @param pads 填充大小，长度为 4，依次表示 [上, 下, 左, 右]
 * @param padding_value 填充使用的数值
 *
 * 填充后张量尺寸变为：
 * rows + pad_rows1 + pad_rows2, cols + pad_cols1 + pad_cols2
 */
template <typename T>
void Tensor<T>::Padding(const std::vector<uint32_t>& pads, T padding_value) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  // 创建新的 Cube 并用 padding_value 填充
  arma::Cube<T> new_data(this->data_.n_rows + pad_rows1 + pad_rows2,
                         this->data_.n_cols + pad_cols1 + pad_cols2, this->data_.n_slices);
  new_data.fill(padding_value);

  // 将原始数据拷贝到新 Cube 的中心区域
  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) = this->data_;
  this->data_ = std::move(new_data);
  this->raw_shapes_ = std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
}

/**
 * @brief 使用指定数值填充整个张量
 * @param value 填充数值
 */
template <typename T>
void Tensor<T>::Fill(T value) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  this->data_.fill(value);
}

/**
 * @brief 使用外部向量填充张量
 * @param values 外部数据向量，长度必须等于张量总元素数
 * @param row_major 若为 true，表示 values 按行主序排列，需要转置后填入各通道
 *
 * 当 row_major 为 true 时，会对每个通道的数据进行转置操作以匹配 arma 的列主序存储。
 */
template <typename T>
void Tensor<T>::Fill(const std::vector<T>& values, bool row_major) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->channels();

    for (uint32_t i = 0; i < channels; ++i) {
      arma::Mat<T> channel_data_t(const_cast<T*>(values.data()) + i * planes, this->cols(),
                                  this->rows(), false, true);
      this->data_.slice(i) = channel_data_t.t();
    }
  } else {
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

/**
 * @brief 打印张量各通道的数据到日志（用于调试）
 */
template <typename T>
void Tensor<T>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

/**
 * @brief 将张量展平为一维
 * @param row_major 若为 true，按行主序展平；否则按内部存储顺序展平
 */
template <typename T>
void Tensor<T>::Flatten(bool row_major) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  const uint32_t size = this->data_.size();
  this->Reshape({size}, row_major);
}

// ------------------- 随机初始化 -------------------

/**
 * @brief 使用正态分布随机填充张量（float 特化）
 * @param mean 均值
 * @param var 标准差
 */
template <>
void Tensor<float>::RandN(float mean, float var) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  std::random_device rd;
  std::mt19937 mt(rd());

  std::normal_distribution<float> dist(mean, var);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}

/**
 * @brief 使用均匀分布随机填充张量（int32_t 特化）
 * @param min 最小值
 * @param max 最大值
 */
template <>
void Tensor<int32_t>::RandU(int32_t min, int32_t max) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  std::random_device rd;
  std::mt19937 mt(rd());

  std::uniform_int_distribution<int32_t> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}

/**
 * @brief 使用均匀分布随机填充张量（uint8_t 特化）
 * @param min 最小值
 * @param max 最大值
 *
 * 由于 MSVC 中 std::uniform_int_distribution<uint8_t> 不被支持，
 * 因此在 MSVC 环境下使用 int32_t 分布并对最大值取模。
 */
template <>
void Tensor<std::uint8_t>::RandU(std::uint8_t min, std::uint8_t max) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  std::random_device rd;
  std::mt19937 mt(rd());

#ifdef _MSC_VER
  std::uniform_int_distribution<int32_t> dist(min, max);
  uint8_t max_value = std::numeric_limits<uint8_t>::max();
  for (uint32_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt) % max_value;
  }
#else
  std::uniform_int_distribution<std::uint8_t> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
#endif
}

/**
 * @brief 使用均匀分布随机填充张量（float 特化）
 * @param min 最小值
 * @param max 最大值
 */
template <>
void Tensor<float>::RandU(float min, float max) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  CHECK(max >= min);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}

/**
 * @brief 将张量所有元素填充为 1
 */
template <typename T>
void Tensor<T>::Ones() {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  this->Fill(T{1});
}

/**
 * @brief 对张量每个元素应用自定义变换函数
 * @param filter 变换函数，签名为 T(T)
 */
template <typename T>
void Tensor<T>::Transform(const std::function<T(T)>& filter) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  this->data_.transform(filter);
}

// ------------------- 形状相关 -------------------

/**
 * @brief 获取张量的原始形状向量
 * @return 原始形状引用，长度范围为 [1, 3]
 *
 * raw_shapes 会省略大小为 1 的维度，例如 {1, 224, 224} 可能记为 {224, 224}。
 */
template <typename T>
const std::vector<uint32_t>& Tensor<T>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  CHECK_LE(this->raw_shapes_.size(), 3);
  CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

/**
 * @brief 改变张量形状，不改变元素总数
 * @param shapes 目标形状，长度不超过 3
 * @param row_major 若为 true，按行主序重新排列数据；否则按内部存储直接 reshape
 *
 * 要求新形状的元素总数必须与原张量一致。
 */
template <typename T>
void Tensor<T>::Reshape(const std::vector<uint32_t>& shapes, bool row_major) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  CHECK(!shapes.empty());
  const size_t origin_size = this->size();
  const size_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), size_t(1), std::multiplies<size_t>());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);
  if (!row_major) {
    // 非行主序：直接使用 arma::Cube 的 reshape
    if (shapes.size() == 3) {
      this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
      this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->data_.reshape(shapes.at(0), shapes.at(1), 1);
      this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
    } else {
      this->data_.reshape(1, shapes.at(0), 1);
      this->raw_shapes_ = {shapes.at(0)};
    }
  } else {
    // 行主序：需要调用 Review 重新排列内存
    if (shapes.size() == 3) {
      this->Review({shapes.at(0), shapes.at(1), shapes.at(2)});
      this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->Review({1, shapes.at(0), shapes.at(1)});
      this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
    } else {
      this->Review({1, 1, shapes.at(0)});
      this->raw_shapes_ = {shapes.at(0)};
    }
  }
}

// ------------------- 原始指针访问 -------------------

/**
 * @brief 获取张量底层数据的原始指针
 * @return 指向首个元素的指针
 */
template <typename T>
T* Tensor<T>::raw_ptr() {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->data_.memptr();
}

/**
 * @brief 获取张量底层数据的常量原始指针
 * @return 指向首个元素的常量指针
 */
template <typename T>
const T* Tensor<T>::raw_ptr() const {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  return this->data_.memptr();
}

/**
 * @brief 获取指定偏移位置的原始指针
 * @param offset 元素偏移量
 * @return 指向 offset 位置元素的指针
 */
template <typename T>
T* Tensor<T>::raw_ptr(size_t offset) {
  const size_t size = this->size();
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;
}

/**
 * @brief 获取指定偏移位置的常量原始指针
 * @param offset 元素偏移量
 * @return 指向 offset 位置元素的常量指针
 */
template <typename T>
const T* Tensor<T>::raw_ptr(size_t offset) const {
  const size_t size = this->size();
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;
}

/**
 * @brief 将张量数据导出为 std::vector
 * @param row_major 若为 true，按行主序导出；否则按内部列主序导出
 * @return 包含张量所有元素的向量
 */
template <typename T>
std::vector<T> Tensor<T>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);
  std::vector<T> values(this->data_.size());

  if (!row_major) {
    // 直接按内部存储顺序拷贝（列主序）
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(), values.begin());
  } else {
    // 按行主序导出：逐个通道转置后拷贝
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      const arma::Mat<T>& channel = this->data_.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}

/**
 * @brief 获取指定通道的原始指针（以平面矩阵方式访问）
 * @param index 通道索引
 * @return 指向该通道首个元素的指针
 */
template <typename T>
T* Tensor<T>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  size_t offset = index * this->plane_size();
  CHECK_LE(offset, this->size());
  T* mem_ptr = this->raw_ptr(offset);
  return mem_ptr;
}

/**
 * @brief 获取指定通道的常量原始指针（以平面矩阵方式访问）
 * @param index 通道索引
 * @return 指向该通道首个元素的常量指针
 */
template <typename T>
const T* Tensor<T>::matrix_raw_ptr(uint32_t index) const {
  CHECK_LT(index, this->channels());
  size_t offset = index * this->plane_size();
  CHECK_LE(offset, this->size());
  const T* mem_ptr = this->raw_ptr(offset);
  return mem_ptr;
}

// ------------------- 内存重排 -------------------

/**
 * @brief 按目标三维形状重新排列张量内存布局
 * @param shapes 目标形状 [channels, rows, cols]，长度必须为 3
 *
 * 该函数用于实现行主序到列主序（或反之）的内存重排，
 * 底层使用 OpenMP 并行加速逐元素映射。
 */
template <typename T>
void Tensor<T>::Review(const std::vector<uint32_t>& shapes) {
  CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
  CHECK_EQ(shapes.size(), 3);
  const uint32_t target_ch = shapes.at(0);
  const uint32_t target_rows = shapes.at(1);
  const uint32_t target_cols = shapes.at(2);

  CHECK_EQ(this->data_.size(), target_ch * target_cols * target_rows);
  arma::Cube<T> new_data(target_rows, target_cols, target_ch);
  const uint32_t plane_size = target_rows * target_cols;

  // 使用 OpenMP 并行化逐通道的数据重排
#pragma omp parallel for
  for (uint32_t channel = 0; channel < this->data_.n_slices; ++channel) {
    const uint32_t plane_start = channel * data_.n_rows * data_.n_cols;
    for (uint32_t src_col = 0; src_col < this->data_.n_cols; ++src_col) {
      const T* col_ptr = this->data_.slice_colptr(channel, src_col);
      for (uint32_t src_row = 0; src_row < this->data_.n_rows; ++src_row) {
        const uint32_t pos_idx = plane_start + src_row * data_.n_cols + src_col;
        const uint32_t dst_ch = pos_idx / plane_size;
        const uint32_t dst_ch_offset = pos_idx % plane_size;
        const uint32_t dst_row = dst_ch_offset / target_cols;
        const uint32_t dst_col = dst_ch_offset % target_cols;
        new_data.at(dst_row, dst_col, dst_ch) = *(col_ptr + src_row);
      }
    }
  }
  this->data_ = std::move(new_data);
}

// ------------------- 模板显式实例化 -------------------

// 显式实例化 float, int32_t, uint8_t 三种类型的 Tensor 模板类
template class Tensor<float>;
template class Tensor<int32_t>;
template class Tensor<uint8_t>;

}  // namespace net_infer
