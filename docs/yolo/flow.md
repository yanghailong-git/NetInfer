这个示例展示了使用 NetInfer 进行 **YOLOv5s 目标检测**的完整流程，分为**数据预处理、模型构建、前向推理、后处理**四个阶段。整体架构和 ResNet 示例一致，核心差异在**预处理（Letterbox）**和**后处理（NMS 解码）**。

---

## 一、整体调用流程图

```
main()
│
├─ 1. 准备数据
│   └─ YoloDemo(image_paths, param_path, bin_path, batch_size=8)
│
├─ 2. 模型构建
│   ├─ RuntimeGraph graph(param_path, bin_path)   // 加载 yolov5s_batch8
│   └─ graph.Build()                              // 拓扑排序 + 创建层
│
├─ 3. 数据预处理（逐图）
│   ├─ cv::imread()                    // 读取原始图像
│   └─ PreProcessImage(image, 640, 640)
│       ├─ Letterbox()                 // 等比缩放 + 灰边填充 → 640×640
│       ├─ cv::cvtColor()              // BGR → RGB
│       ├─ cv::split()                 // 按通道拆分
│       ├─ std::make_shared<Tensor>()  // 创建 Tensor [3,640,640]
│       └─ memcpy()                    // HWC → CHW
│
├─ 4. 前向推理
│   ├─ graph.set_inputs("pnnx_input_0", inputs)   // 注入 batch=8 的张量
│   ├─ graph.Forward(true)                        // 执行推理（debug模式）
│   └─ graph.get_outputs("pnnx_output_0")         // 获取输出
│
└─ 5. 后处理（逐图解码）
    ├─ 遍历 25200 个 anchor 预测
    │   ├─ 解析中心点 (x,y)、宽高 (w,h)、目标置信度 conf
    │   ├─ 遍历 80 个类别找 best_class_id / best_conf
    │   └─ 若 conf ≥ 0.25，存入候选框列表
    ├─ cv::dnn::NMSBoxes()             // NMS 非极大值抑制
    ├─ ScaleCoords()                   // 坐标映射回原图尺寸
    └─ cv::rectangle / cv::putText     // 画框保存结果图
```

---

## 二、和 ResNet 示例的核心差异

| 阶段 | ResNet | YOLO |
|------|--------|------|
| **预处理** | `cv::resize` 直接拉伸 + ImageNet 标准化 | `Letterbox` 等比缩放 + 灰边填充（保持长宽比） |
| **Batch** | 1 | 8（模型文件为 `yolov5s_batch8`） |
| **输出维度** | `(1, 1000)` logits | `(8, 25200, 85)` 检测头输出 |
| **后处理** | `SoftmaxLayer` + 找最大索引 | 手动解析 bbox + `NMSBoxes` + `ScaleCoords` |
| **输出用途** | 图像分类（Top-1 类别） | 目标检测（画 bounding box） |

---

## 三、关键类与函数的详细交互

### 1. 预处理：`Letterbox` vs `Resize`

YOLO 为了保持目标长宽比、避免形变，使用 **Letterbox** 处理：

```cpp
Letterbox(image, out_image, {640, 640}, 32, {114,114,114}, true);
```

内部流程：
1. 计算缩放比例 `r = min(640/H, 640/W)`
2. 等比缩放图像（如 1920×1080 → 640×360）
3. 用灰色 `(114,114,114)` 填充上下/左右黑边，使最终尺寸为 **640×640**（且是 stride=32 的整数倍）

> ResNet 是直接 `cv::resize` 拉伸到 224×224；YOLO 是等比缩放+padding，这是检测任务的标准做法。

---

### 2. 模型推理：`RuntimeGraph` 的执行链路

和 ResNet **完全一致**：

```
RuntimeGraph::Build()
    ├─ Init()                     // pnnx::Graph::load() 解析 .param + .bin
    ├─ CreateNodeRelation()       // 建立拓扑关系
    │   └─ LayerRegisterer::CreateLayer()  // 工厂创建 YoloDetectLayer/Conv/ReLU 等
    ├─ ReverseTopoSort()          // 确定执行顺序
    └─ InitOperatorInput/Output() // 分配张量内存

graph.Forward(true)               // 按拓扑序逐层 Forward()
    ├─ ExecuteLayer()             // Conv/ReLU/...
    └─ PropagateLayerOutputs()    // 指针传播
```

> YOLOv5s 的模型结构中包含 `YoloDetectLayer`（检测头层），负责将三个尺度的特征图解码为最终的检测框预测。最终在 `pnnx_output_0` 输出形状为 `(batch, 25200, 85)` 的张量。

---

### 3. 后处理：从张量到检测框

YOLO 后处理全部在 `yolo_test.cpp` 中手动完成，**没有调用框架内置层**：

#### 3.1 解码预测结果

输出张量形状：`(batch=8, elements=25200, num_info=85)`

每个 anchor 的 85 个值含义：
- `[0]` `x`：检测框中心点 x
- `[1]` `y`：检测框中心点 y
- `[2]` `w`：检测框宽度
- `[3]` `h`：检测框高度
- `[4]` `obj_conf`：目标置信度（是否包含物体）
- `[5:85]` `cls_conf`：80 个 COCO 类别的置信度

```cpp
for (uint32_t e = 0; e < elements; ++e) {
    float cls_conf = output->at(b, e, 4);          // 目标置信度
    if (cls_conf >= conf_thresh) {                 // 0.25 阈值过滤
        int center_x = output->at(b, e, 0);
        int center_y = output->at(b, e, 1);
        int width    = output->at(b, e, 2);
        int height   = output->at(b, e, 3);
        
        // 找最佳类别
        int best_class_id = -1;
        float best_conf = -1.f;
        for (uint32_t j = 5; j < num_info; ++j) {
            if (output->at(b, e, j) > best_conf) {
                best_conf = output->at(b, e, j);
                best_class_id = j - 5;
            }
        }
        
        boxes.push_back({left, top, width, height});
        confs.push_back(best_conf * cls_conf);      // 综合置信度 = 类别置信度 × 目标置信度
        class_ids.push_back(best_class_id);
    }
}
```

#### 3.2 NMS 非极大值抑制

```cpp
cv::dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);
```

- **输入**：所有超过置信度阈值的候选框
- **原理**：对重叠度（IoU）超过 `0.25` 的框，只保留置信度最高的那个，去除冗余检测框

#### 3.3 坐标映射回原图：`ScaleCoords`

```cpp
ScaleCoords(cv::Size{640, 640}, det.box, cv::Size{origin_w, origin_h});
```

因为预处理时用了 Letterbox（等比缩放+padding），检测框坐标是在 **640×640 填充后图像**上的。`ScaleCoords` 负责：
1. 减去灰边 padding 的偏移量
2. 除以缩放比例 `gain`
3. 映射回原始图像坐标系

---

## 四、数据流张量形状变化

| 阶段 | 张量形状 | 说明 |
|------|---------|------|
| 原始图像 | `(H, W, 3)` | OpenCV BGR，尺寸各异 |
| Letterbox 后 | `(640, 640, 3)` | 等比缩放 + 灰边填充 |
| Tensor 输入 | `(3, 640, 640)` | CHW 格式，归一化到 `[0,1]` |
| 模型输入 | `(8, 3, 640, 640)` | batch=8 |
| 模型输出 | `(8, 25200, 85)` | YOLOv5 检测头输出 |
| NMS 后 | N 个 `Detection` | 每张图保留有效检测框 |

---

## 五、总结一句话

> `yolo_test.cpp` 的执行流程就是：**用 `Letterbox` 将多张图像预处理为 640×640 的 batch 张量 → `RuntimeGraph` 执行 YOLOv5s 前向推理 → 对输出张量 `(batch, 25200, 85)` 逐 anchor 解码 bbox/置信度/类别 → `NMSBoxes` 去重 → `ScaleCoords` 映射回原图 → OpenCV 画框保存结果。**