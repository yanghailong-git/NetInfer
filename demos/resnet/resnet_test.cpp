#include <algorithm>
#include <cassert>
#include <opencv2/opencv.hpp>

#include "../source/layer/details/softmax.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

// python ref https://pytorch.org/hub/pytorch_vision_resnet/

/**
 * @brief 对输入图像进行预处理，转换为模型所需的张量格式
 * @param image 输入的OpenCV图像(BGR格式)
 * @return 预处理后的浮点张量，形状为 [3, 224, 224]
 *
 * 预处理流程：
 * 1. 将图像resize到 224x224
 * 2. BGR -> RGB
 * 3. 转换为 float32 类型
 * 4. 按通道拆分
 * 5. 将数据拷贝到张量中（注意opencv是hwc，张量是chw）
 * 6. 归一化到 [0,1]（除以255）
 * 7. ImageNet 标准化：(x - mean) / std
 */
net_infer::sftensor PreProcessImage(const cv::Mat& image) {
  using namespace net_infer;
  assert(!image.empty());

  // 1. 调整输入图像大小为 224x224
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(224, 224));

  // 2. 将 BGR 格式转换为 RGB 格式
  cv::Mat rgb_image;
  cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

  // 3. 转换为 32位浮点型 3通道图像
  rgb_image.convertTo(rgb_image, CV_32FC3);

  // 4. 将多通道图像按通道拆分为多个单通道图像
  std::vector<cv::Mat> split_images;
  cv::split(rgb_image, split_images);

  // 5. 定义输入张量的维度：3通道，224高，224宽
  uint32_t input_w = 224;
  uint32_t input_h = 224;
  uint32_t input_c = 3;
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  // 6. 将拆分后的各通道数据拷贝到张量中
  //    OpenCV 的 Mat 是按 H×W 存储的，与张量 slice 的内存布局一致
  uint32_t index = 0;
  for (const auto& split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat& split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
  }

  // 7. ImageNet 预训练模型使用的均值和标准差
  float mean_r = 0.485f;
  float mean_g = 0.456f;
  float mean_b = 0.406f;

  float var_r = 0.229f;
  float var_g = 0.224f;
  float var_b = 0.225f;

  // 8. 归一化与标准化处理
  assert(input->channels() == 3);
  // 将像素值从 [0, 255] 缩放到 [0, 1]
  input->data() = input->data() / 255.f;
  // 对每个通道进行标准化：(value - mean) / std
  input->slice(0) = (input->slice(0) - mean_r) / var_r;
  input->slice(1) = (input->slice(1) - mean_g) / var_g;
  input->slice(2) = (input->slice(2) - mean_b) / var_b;

  return input;
}

/**
 * @brief ResNet 图像分类推理示例程序
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组，argv[1] 为待推理的图像路径
 * @return 程序退出码，0 表示正常退出
 *
 * 使用流程：
 * 1. 读取输入图像
 * 2. 图像预处理
 * 3. 加载模型参数与权重，构建计算图
 * 4. 执行前向推理
 * 5. Softmax 后处理
 * 6. 输出概率最大的类别及其置信度
 */
int main(int argc, char* argv[]) {
  // 检查命令行参数，要求传入图像路径
  if (argc != 2) {
    printf("usage: ./resnet_test [image path]\n");
    exit(-1);
  }
  using namespace net_infer;

  // 获取输入图像路径
  const std::string& path = argv[1];

  // 1. 准备输入数据：batch_size 设为 1
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    // 使用 OpenCV 读取图像（默认读取为 BGR 格式）
    cv::Mat image = cv::imread(path);
    // 对图像进行预处理
    sftensor input = PreProcessImage(image);
    inputs.push_back(input);
  }

  // 2. 指定模型参数文件和权重文件路径
  const std::string& param_path =
      "/workspace/neural_infer/NetInfer/models/resnet/"
      "resnet18_batch1.pnnx.param";
  const std::string& weight_path =
      "/workspace/neural_infer/NetInfer/models/resnet/"
      "resnet18_batch1.pnnx.bin";

  // 3. 构建计算图并加载模型
  RuntimeGraph graph(param_path, weight_path);
  graph.Build();
  // 将预处理后的输入数据设置到计算图的输入节点
  graph.set_inputs("pnnx_input_0", inputs);

  // 4. 执行前向推理，并统计推理耗时
  TICK(forward)
  graph.Forward(false);
  // 获取计算图的输出结果
  std::vector<std::shared_ptr<Tensor<float>>> outputs =
      graph.get_outputs("pnnx_output_0");
  TOCK(forward)

  assert(outputs.size() == batch_size);

  // 5. 对模型原始输出应用 Softmax，将 logits 转换为概率分布
  std::vector<sftensor> outputs_softmax(batch_size);
  SoftmaxLayer softmax_layer(0);
  softmax_layer.Forward(outputs, outputs_softmax);
  assert(outputs_softmax.size() == batch_size);

  // 6. 遍历每个输出，找出概率最大的类别索引及对应概率
  for (int i = 0; i < outputs_softmax.size(); ++i) {
    const sftensor& output_tensor = outputs_softmax.at(i);
    // ResNet18 在 ImageNet 上输出 1000 个类别的分数
    assert(output_tensor->size() == 1 * 1000);

    // 找到类别概率最大的种类
    float max_prob = -1;
    int max_index = -1;
    for (int j = 0; j < output_tensor->size(); ++j) {
      float prob = output_tensor->index(j);
      if (max_prob <= prob) {
        max_prob = prob;
        max_index = j;
      }
    }
    // 打印置信度最高的类别索引及其概率
    printf("class with max prob is %f index %d\n", max_prob, max_index);
  }

  return 0;
}
