#ifndef NET_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#include "convolution.hpp"
#include "layer/abstract/layer.hpp"

namespace net_infer {
// YOLO detection head layer.
// It applies 1x1 convolutions to multi-scale feature maps, decodes bounding
// boxes using anchor grids and strides, and concatenates the results.
class YoloDetectLayer : public Layer<float> {
 public:
  explicit YoloDetectLayer(int32_t stages, int32_t num_classes, int32_t num_anchors,
                           std::vector<float> strides, std::vector<arma::fmat> anchor_grids,
                           std::vector<arma::fmat> grids,
                           std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers);

  // Forward pass: decode predictions and concatenate stage outputs.
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // Factory method to create a YoloDetectLayer from runtime attributes.
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& yolo_detect_layer);

 private:
  int32_t stages_ = 0;               // Number of detection stages (e.g., 3 for YOLOv5).
  int32_t num_classes_ = 0;          // Number of object classes.
  int32_t num_anchors_ = 0;          // Number of anchors per grid cell.
  std::vector<float> strides_;       // Stride for each stage.
  std::vector<arma::fmat> anchor_grids_; // Anchor grid matrices.
  std::vector<arma::fmat> grids_;        // Grid offset matrices.
  std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers_; // 1x1 conv layers per stage.
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
