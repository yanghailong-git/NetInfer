这个示例展示了使用 NetInfer 进行 **ResNet18 图像分类**的完整流程，分为**数据预处理、模型构建、前向推理、后处理**四个阶段。

---

## 一、整体调用流程图

```
main()
│
├─ 1. 数据准备
│   ├─ cv::imread()                    // OpenCV 读取图像 (BGR)
│   └─ PreProcessImage()               // 预处理为 [3,224,224] 张量
│       ├─ cv::resize()                // 224x224
│       ├─ cv::cvtColor()              // BGR → RGB
│       ├─ cv::split()                 // 按通道拆分
│       ├─ std::make_shared<ftensor>() // 创建 Tensor [C,H,W]
│       ├─ memcpy()                    // HWC → CHW 数据拷贝
│       └─ 归一化 / ImageNet 标准化     // /255, (x-mean)/std
│
├─ 2. 模型构建
│   ├─ RuntimeGraph graph(param, bin)  // 构造计算图
│   └─ graph.Build()                   // 构建图
│       ├─ Init()                      // 加载 PNNX 模型
│       │   ├─ pnnx::Graph::load()     // 解析 .param + .bin
│       │   ├─ InitGraphOperatorsInput()   // 提取输入形状/类型
│       │   ├─ InitGraphOperatorsOutput()  // 提取输出消费者
│       │   ├─ InitGraphAttrs()        // 提取权重数据
│       │   └─ InitGraphParams()       // 提取超参数 (stride/padding等)
│       ├─ CreateNodeRelation()        // 建立节点拓扑关系
│       │   └─ CreateLayer()           // LayerRegisterer 工厂创建层
│       ├─ ReverseTopoSort()           // 反向拓扑排序确定执行顺序
│       └─ InitOperatorInput/Output()  // 初始化张量内存空间
│
├─ 3. 前向推理
│   ├─ graph.set_inputs(name, inputs)  // 将输入张量注入计算图
│   └─ graph.Forward(false)            // 执行推理
│       └─ 遍历 operators_ (拓扑序)
│           ├─ 跳过 pnnx.Input/Output 占位算子
│           ├─ ExecuteLayer()          // 执行单层
│           │   └─ layer->Forward()    // 调用具体层的前向 (Conv/ReLU/...)
│           └─ PropagateLayerOutputs() // 输出传播给后继节点
│
├─ 4. 后处理
│   ├─ graph.get_outputs(name)         // 获取推理结果 [1,1000]
│   ├─ SoftmaxLayer::Forward()         // Softmax 转概率
│   └─ 遍历找 max_index / max_prob     // 输出 Top-1 类别
│
```

---

## 二、关键类与函数的交互细节

### 1. RuntimeGraph — 计算图 orchestrator

这是整个推理的**总指挥**，内部维护三个核心列表：

```cpp
class RuntimeGraph {
  std::vector<std::shared_ptr<RuntimeOperator>> input_ops_;   // 输入占位节点
  std::vector<std::shared_ptr<RuntimeOperator>> output_ops_;  // 输出占位节点
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;   // 所有算子（拓扑序）
};
```

#### `Build()` 内部的三件事

| 步骤 | 调用的函数 | 作用 |
|------|-----------|------|
| 加载模型 | `Init()` | 用 `pnnx::Graph::load()` 解析 `.param` 拓扑 + `.bin` 权重 |
| 建关系 | `CreateNodeRelation()` | 根据 `output_names` 把算子串成图；通过 **Layer 工厂** (`LayerRegisterer::CreateLayer`) 为每个算子创建对应的 `Layer` 对象 |
| 定顺序 | `ReverseTopoSort()` | 从输出端反向 DFS，给每个算子分配 `start_time`，确保执行时输入先算、输出后算 |

> `CreateLayer()` 就是之前解释过的**懒加载工厂模式**：根据 `op->type`（如 `"nn.Conv2d"`）从全局注册表找到对应的创建函数，生成 `ConvolutionLayer`、`ReLULayer` 等具体对象。

---

### 2. 数据如何在算子间流动

每个 `RuntimeOperator` 都有 `input_operands` 和 `output_operands`：

```
输入图像 → [input_op] ── datas ──→ [Conv2d] ── datas ──→ [ReLU] ── datas ──→ [output_op]
              ↑                        ↑                      ↑
         input_operands          input_operands         input_operands
         output_operands          output_operands        output_operands
```

`Forward()` 中的核心循环：

```cpp
for (const auto& current_op : operators_) {
    // 1. 执行当前层
    StatusCode status = ExecuteLayer(current_op->layer, ...);
    
    // 2. 将当前层输出张量指针"赋值"给后继层的输入
    PropagateLayerOutputs(current_op, current_op->output_operands->datas);
}
```

**注意**：`PropagateLayerOutputs` 做的是**指针赋值（`std::shared_ptr` 拷贝）**，而不是深拷贝数据。这样内存效率高。

---

### 3. 层的注册与创建链路

以 `nn.Conv2d` 为例，层的创建链路如下：

```
resnet_test.cpp
    └─ RuntimeGraph::Build()
        └─ CreateNodeRelation()
            └─ RuntimeGraph::CreateLayer(op)
                └─ LayerRegisterer::CreateLayer(op)   ← 工厂方法
                    └─ registry["nn.Conv2d"]()        ← 查找注册表
                        └─ ConvolutionLayer::CreateInstance(op, layer)
                            └─ new ConvolutionLayer(...)  ← 具体层对象
```

`LayerRegisterer` 的全局注册表在程序启动时由各个层的**静态注册对象**自动填充（如 `source/layer/details/convolution.cpp` 中的 `LayerRegistererWrapper`）。

---

### 4. 内存复用机制

`Build()` 中有一行：

```cpp
RuntimeOperatorUtils<float>::InitOperatorOutput(
    graph_->ops, operators_, enable_memory_reuse_);
```

如果 `enable_memory_reuse_ = true`（默认），框架会根据拓扑排序计算的 `end_time`（某层输出被最后使用的时刻），**复用已失效的中间张量内存**，减少显存/内存占用。

---

## 三、总结一句话

> `resnet_test.cpp` 的执行流程就是：**用 `RuntimeGraph` 加载并编译 PNNX 模型为拓扑有序的计算图 → 将 OpenCV 图像预处理成 `Tensor` 喂给输入节点 → 按拓扑序逐层调用 `Layer::Forward()` 并传播张量 → 最后从输出节点取结果做 Softmax 分类。**