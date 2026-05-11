
---

## 类图要点说明

### 1. 继承体系

| 分支 | 说明 |
|------|------|
| **`Layer<float>`** | 所有算子的根基类，定义 `Forward()` / `Check()` / `set_runtime_operator()` |
| **`ParamLayer`** | 有参数层（Conv / Linear / BN / Matmul），管理 `weights_` 和 `bias_` |
| **`BaseConvolutionLayer`** | 卷积基类，封装 padding/stride/dilation/groups 等卷积通用属性 |
| **`ActivationLayer`** | 激活函数基类，所有激活函数（ReLU/SiLU/Sigmoid 等）均继承自此 |

### 2. 组合关系

| 容器 | 成员 | 关系 |
|------|------|------|
| `RuntimeGraph` | `vector<RuntimeOperator>` | 1 对 n 强组合 |
| `RuntimeOperator` | `shared_ptr<Layer>` | 1 对 1 关联 |
| `RuntimeOperator` | `vector<RuntimeOperand>` | 1 对 n 强组合 |
| `RuntimeOperand` | `vector<shared_ptr<Tensor>>` | 1 对 n 强组合 |
| `pnnx::Graph` | `vector<pnnx::Operator*>` | 1 对 n 强组合 |

### 3. 关键设计模式在 UML 中的体现

- **单例**：`LayerRegisterer::registry_`（懒加载，`{static}` 修饰）
- **工厂**：`LayerRegisterer::CreateLayer()` 根据 `op->type` 动态创建 `Layer` 子类
- **模板方法**：`Layer::Forward()` 定义接口，子类（`ConvolutionLayer`、`SoftmaxLayer` 等）实现具体逻辑