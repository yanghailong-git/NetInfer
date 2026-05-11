#ifndef NET_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#include "convolution.hpp"
#include "layer/abstract/layer.hpp"

namespace net_infer {
// YOLO 检测头层。
// 对多尺度特征图应用 1x1 卷积，使用锚框网格和步长解码边界框，并拼接结果。
class YoloDetectLayer : public Layer<float> {
 public:
  explicit YoloDetectLayer(int32_t stages, int32_t num_classes, int32_t num_anchors,
                           std::vector<float> strides, std::vector<arma::fmat> anchor_grids,
                           std::vector<arma::fmat> grids,
                           std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers);

  // 前向传播：解码预测结果并拼接各阶段输出。
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 工厂方法，根据运行时属性创建 YoloDetectLayer。
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& yolo_detect_layer);

 private:
  int32_t stages_ = 0;               // 检测阶段数（例如 YOLOv5 为 3）。
  int32_t num_classes_ = 0;          // 目标类别数。
  int32_t num_anchors_ = 0;          // 每个网格单元的锚框数。
  std::vector<float> strides_;       // 每个阶段的步长。
  std::vector<arma::fmat> anchor_grids_; // 锚框网格矩阵。
  std::vector<arma::fmat> grids_;        // 网格偏移矩阵。
  std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers_; // 每个阶段的 1x1 卷积层。
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
