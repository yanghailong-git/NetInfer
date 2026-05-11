这个示例展示了使用 NetInfer 进行 **MobileNetV2 图像分类**的完整流程。整体调用链路和 ResNet18 示例**几乎完全一致**，核心差异仅在于**模型文件**和**内存复用策略**的显式关闭。

---

## 一、整体调用流程图

```
main()
│
├─ 1. 数据准备
│   ├─ cv::imread()                    // OpenCV 读取图像 (BGR)
│   └─ PreProcessImage(image)          // 预处理为 [3,224,224] 张量
│       ├─ cv::resize()                // 224×224
│       ├─ cv::cvtColor()              // BGR → RGB
│       ├─ cv::split()                 // 按通道拆分
│       ├─ std::make_shared<ftensor>() // 创建 Tensor [C,H,W]
│       ├─ memcpy()                    // HWC → CHW
│       └─ /255 + ImageNet 标准化      // (x-mean)/std
│
├─ 2. 模型构建
│   ├─ RuntimeGraph graph(param, bin)  // 加载 mobilenet_v2
│   ├─ graph.set_enable_memory_reuse(false)  // ← 关键差异：关闭内存复用
│   └─ graph.Build()                   // 拓扑排序 + 创建层 + 初始化内存
│       ├─ Init()                      // pnnx::Graph::load() 解析 .param + .bin
│       ├─ CreateNodeRelation()        // 建图 + LayerRegisterer 工厂创建层
│       ├─ ReverseTopoSort()           // 反向拓扑排序定执行顺序
│       └─ InitOperatorInput/Output()  // 分配张量空间
│
├─ 3. 前向推理
│   ├─ graph.set_inputs("pnnx_input_0", inputs)
│   ├─ TICK(forward)
│   ├─ graph.Forward(false)            // 按拓扑序逐层执行
│   │   ├─ ExecuteLayer()              // Conv/DepthwiseConv/ReLU6/...
│   │   └─ PropagateLayerOutputs()     // 指针传播输出到后继节点
│   ├─ outputs = graph.get_outputs("pnnx_output_0")
│   └─ TOCK(forward)
│
├─ 4. 后处理
│   ├─ SoftmaxLayer::Forward()         // Softmax 转概率
│   └─ 遍历找 max_index / max_prob     // Top-1 类别输出
│
```

---

## 二、和 ResNet 示例的核心差异

| 对比项 | ResNet18 | MobileNetV2 |
|--------|----------|-------------|
| **模型文件** | `resnet18_batch1.pnnx.param/bin` | `mobilenet_v2.pnnx.param/bin` |
| **核心算子** | Conv2d + ReLU + BasicBlock | Conv2d + DepthwiseConv + ReLU6 + InvertedResidual |
| **内存复用** | 默认开启 (`true`) | **显式关闭 (`false`)** |
| **batch_size** | 1 | 1 |
| **后处理** | Softmax + 找 Top-1 | Softmax + 找 Top-1 |

---

## 三、关键代码解析

### 1. 为什么关闭内存复用？

```cpp
graph.set_enable_memory_reuse(false);
```

这是 MobileNetV2 示例中**唯一和 ResNet 不同的代码行**。

**原因**：MobileNetV2 的 Inverted Residual 块包含**残差连接（skip connection）**，且网络结构比 ResNet 更"窄"更"深"，张量复用的生命周期管理更复杂。如果开启内存复用，某些中间张量可能在后续节点还需要时被提前覆盖，导致**use-after-free**。

> `runtime_ir.hpp` 中的注释也说明了这一点：
> > *"Disable this for models with complex residual connections (e.g. MobilenetV2) where shared raw memory can cause use-after-free."*

关闭后，每个算子的输入/输出张量都独立分配，牺牲少量内存换取绝对安全。

---

### 2. MobileNetV2 特有的层类型

虽然调用代码和 ResNet 一样，但 `LayerRegisterer` 工厂在 `Build()` 阶段实际创建的层类型不同：

| MobileNetV2 层 | 对应框架类 | 作用 |
|---------------|-----------|------|
| 标准卷积 | `ConvolutionLayer` | 1×1 点卷积 + 3×3 首卷积 |
| Depthwise 卷积 | `ConvolutionLayer` (groups=in_ch) | 逐通道 3×3 卷积 |
| Inverted Residual | 由多个 Conv + ReLU6 + `Expression(add)` 组合 | 线性瓶颈 + 残差连接 |
| ReLU6 | `ReLU6Layer` | `min(max(x,0),6)`，量化友好 |
| 全局平均池化 | `AdaptiveAvgPoolingLayer` | 7×7 → 1×1 |
| 分类头 | `LinearLayer` | 1280 → 1000 |

---

### 3. 完整的 `RuntimeGraph` 执行链路（和 ResNet 一致）

```
RuntimeGraph::Build()
    ├─ Init()
    │   └─ pnnx::Graph::load(param, bin)     // 解析 PNNX 模型
    ├─ CreateNodeRelation()
    │   └─ RuntimeGraph::CreateLayer(op)
    │       └─ LayerRegisterer::CreateLayer(op)   // 工厂方法
    │           └─ registry["nn.Conv2d"]() / ["nn.ReLU6"]() / ...
    │               └─ 各 Layer::CreateInstance()   // 初始化参数和权重
    ├─ ReverseTopoSort()                      // DFS 定执行顺序
    └─ InitOperatorInput/Output()             // 分配内存（不复用）

graph.Forward(false)
    └─ for each op in operators_ (拓扑序)
        ├─ ExecuteLayer(layer, name, type, debug)
        │   └─ layer->Forward()               // 实际计算
        └─ PropagateLayerOutputs()            // shared_ptr 传播
```

---

## 四、总结一句话

> `mobile.cpp` 的执行流程和 ResNet 完全相同：**预处理 → `RuntimeGraph` 加载 MobileNetV2 → `Build()` 构建拓扑图 → `Forward()` 逐层推理 → Softmax → Top-1 输出**。唯一区别在于显式关闭了**内存复用**，以避免 MobileNetV2 复杂残差连接导致的内存安全问题。