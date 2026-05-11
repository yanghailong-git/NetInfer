#include "runtime/runtime_ir.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "utils/time/time_logging.hpp"

namespace net_infer {

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_enable_memory_reuse(bool enable) {
  enable_memory_reuse_ = enable;
}

void RuntimeGraph::set_bin_path(const std::string& bin_path) { this->bin_path_ = bin_path; }

void RuntimeGraph::set_param_path(const std::string& param_path) { this->param_path_ = param_path; }

const std::string& RuntimeGraph::param_path() const { return this->param_path_; }

const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }

/**
 * @brief 判断是否为量化算子（当前未支持）
 * @return 固定返回 false
 */
static bool IsQuantizeOp(const pnnx::Operator* op) { return false; }

/**
 * @brief 从 pnnx 模型文件加载并初始化运行时算子
 * @return 加载成功返回 true，否则返回 false
 *
 * 加载流程：
 * 1. 使用 pnnx::Graph 加载 .param 和 .bin 文件
 * 2. 遍历所有算子，跳过量化算子
 * 3. 对每个算子提取：名称、类型、输入、输出、属性（权重）、参数
 * 4. 将所有运行时算子存入 operators_ 列表
 */
bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int32_t load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Can not find the param path or bin path: " << param_path_ << " " << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator*> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  operators_.clear();
  for (const pnnx::Operator* op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node in the model";
      continue;
    } else {
      if (!IsQuantizeOp(op)) {
        std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
        // 初始化算子的名称
        runtime_operator->name = op->name;
        runtime_operator->type = op->type;

        // 初始化算子中的input
        InitGraphOperatorsInput(op->inputs, runtime_operator);

        // 记录输出operand中的名称
        InitGraphOperatorsOutput(op->outputs, runtime_operator);

        // 初始化算子中的attribute(权重)
        InitGraphAttrs(op->attrs, runtime_operator);

        // 初始化算子中的parameter
        InitGraphParams(op->params, runtime_operator);
        this->operators_.push_back(runtime_operator);
      } else {
        LOG(FATAL) << "UnSupported quantize operator in the model " << op->name
                   << " type: " << op->type;
      }
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

/**
 * @brief 构建计算图，完成拓扑排序和内存初始化
 *
 * 构建流程：
 * 1. 若尚未 Init，先调用 Init()
 * 2. 构建算子间的连接关系（CreateNodeRelation）
 * 3. 反向拓扑排序（ReverseTopoSort），确定执行顺序
 * 4. 初始化所有算子的输入/输出数据空间
 * 5. 释放 pnnx::Graph 对象（不再需要）
 */
void RuntimeGraph::Build() {
  if (graph_state_ == GraphState::Complete) {
    LOG(INFO) << "Model has been built already!";
    return;
  }

  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph || graph_state_ == GraphState::NeedInit) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int32_t(graph_state_);
  LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty, may be no init";

  // 构建节点关系
  CreateNodeRelation();

  // 节点拓扑排序
  ReverseTopoSort();

  // 初始化节点的输入和输出空间
  RuntimeOperatorUtils<float>::InitOperatorInput(operators_);
  RuntimeOperatorUtils<float>::InitOperatorOutput(graph_->ops, operators_, enable_memory_reuse_);

  graph_state_ = GraphState::Complete;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

/**
 * @brief 执行单个层的前向推理
 * @param layer 待执行的层对象
 * @param op_name 算子名称（用于计时日志）
 * @param op_type 算子类型（用于计时日志）
 * @param is_debug 若为 true，则记录该层的执行耗时
 * @return 执行状态码
 */
template <typename T>
StatusCode ExecuteLayer(const std::shared_ptr<Layer<T>>& layer, const std::string& op_name,
                        const std::string& op_type, bool is_debug) {
  CHECK(layer != nullptr);
  StatusCode status;
  if (is_debug) {
    utils::LayerTimeLogging layer_time_logging(op_name, op_type);
    status = layer->Forward();
  } else {
    status = layer->Forward();
  }
  return status;
}

/**
 * @brief 执行整个计算图的前向推理
 * @param debug 若为 true，则输出各层执行时间的汇总日志
 *
 * 推理流程：
 * 1. 检查图是否已构建完成
 * 2. 按拓扑顺序遍历所有算子
 * 3. 跳过输入/输出占位算子（pnnx.Input / pnnx.Output）
 * 4. 对每个算子调用 ExecuteLayer 进行前向计算
 * 5. 将当前层的输出传播到后继算子的输入（PropagateLayerOutputs）
 * 6. 检查所有算子是否都已执行
 */
void RuntimeGraph::Forward(bool debug) {
  // 检查当前的执行图是否已经初始化完毕
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!"
               << ", current state is " << int32_t(graph_state_);
  }

  if (debug) {
    utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
  }

  for (const auto& current_op : operators_) {
    current_op->has_forward = false;
    CHECK_GT(current_op->start_time, 0);

    // 输入/输出占位算子不需要执行层计算
    if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
      current_op->has_forward = true;
      continue;
    }

    CHECK(current_op->layer != nullptr)
        << "The layer corresponding to the op " << current_op->name
        << " is empty, indicating that it may not have been created.";

    StatusCode status = ExecuteLayer(current_op->layer, current_op->name, current_op->type, debug);
    CHECK(status == StatusCode::kSuccess)
        << current_op->layer->layer_name()
        << " layer forward failed, error code: " << int32_t(status);

    current_op->has_forward = true;
    // 将当前层的输出张量传播给所有后继算子
    PropagateLayerOutputs(current_op, current_op->output_operands->datas);
  }

  if (debug) {
    utils::LayerTimeLogging::SummaryLogging();
  }

  // 断言所有算子都已完成前向传播
  for (const auto& op : operators_) {
    LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
  }
}

/**
 * @brief 根据运行时算子创建对应的层对象
 * @param op 运行时算子
 * @return 创建的层对象
 */
template <typename T>
std::shared_ptr<Layer<T>> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperatorBase<T>>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  auto layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}

/**
 * @brief 将 pnnx 的输入操作数转换为运行时输入操作数
 * @param inputs pnnx 输入操作数列表
 * @param runtime_operator 目标运行时算子
 *
 * 提取每个输入的形状、生产者名称、数据类型，并存入 runtime_operator 的 input_operands。
 */
template <typename T>
void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (inputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* input : inputs) {
    if (!input) {
      continue;
    }

    std::vector<int32_t> dims;
    const pnnx::Operator* producer = input->producer;

    for (int32_t dim : input->shape) {
      dims.push_back(dim);
    }
    CHECK(!dims.empty());
    std::shared_ptr<RuntimeOperandBase<T>> runtime_operand =
        std::make_shared<RuntimeOperandBase<T>>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = dims;
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 7: {
        runtime_operand->type = RuntimeDataType::kTypeInt8;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }
  }
}

/**
 * @brief 记录算子的输出操作数对应的消费者名称
 * @param outputs pnnx 输出操作数列表
 * @param runtime_operator 目标运行时算子
 */
template <typename T>
void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (outputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* output : outputs) {
    if (!output) {
      continue;
    }
    const auto& consumers = output->consumers;
    for (const auto& c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

/**
 * @brief 将 pnnx 的参数转换为运行时参数
 * @param params pnnx 参数映射表
 * @param runtime_operator 目标运行时算子
 *
 * 支持的参数类型：bool、int、float、string、int_array、float_array、string_array
 */
template <typename T>
void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (params.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const auto& [name, parameter] : params) {
    const int32_t type = parameter.type;
    switch (type) {
      case int32_t(RuntimeParameterType::kParameterUnknown): {
        std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterBool): {
        std::shared_ptr<RuntimeParameterBool> runtime_parameter =
            std::make_shared<RuntimeParameterBool>(parameter.b);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterInt): {
        std::shared_ptr<RuntimeParameterInt> runtime_parameter =
            std::make_shared<RuntimeParameterInt>(parameter.i);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterFloat): {
        std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
            std::make_shared<RuntimeParameterFloat>(parameter.f);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterString): {
        std::shared_ptr<RuntimeParameterString> runtime_parameter =
            std::make_shared<RuntimeParameterString>(parameter.s);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterIntArray): {
        std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
            std::make_shared<RuntimeParameterIntArray>(parameter.ai);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterFloatArray): {
        std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
            std::make_shared<RuntimeParameterFloatArray>(parameter.af);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int32_t(RuntimeParameterType::kParameterStringArray): {
        std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
            std::make_shared<RuntimeParameterStringArray>(parameter.as);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type: " << type;
      }
    }
  }
}

/**
 * @brief 将 pnnx 的属性（权重）转换为运行时属性
 * @param attrs pnnx 属性映射表
 * @param runtime_operator 目标运行时算子
 *
 * 目前仅支持 float32 类型的属性。
 */
template <typename T>
void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
                                  const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (attrs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const auto& [name, attr] : attrs) {
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>(
            attr.shape, RuntimeDataType::kTypeFloat32, attr.data);
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown attribute type: " << attr.type;
      }
    }
  }
}

/**
 * @brief 将当前算子的输出传播到所有后继算子的输入
 * @param current_op 当前算子
 * @param layer_output_datas 当前算子的输出张量列表
 *
 * 遍历 current_op 的所有后继算子，找到对应于 current_op 的输入操作数，
 * 并将输出张量赋值给该输入操作数的 datas。
 */
template <typename T>
void RuntimeGraph::PropagateLayerOutputs(
    const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
    const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas) {
  // For each next operator of current operator
  for (const auto& [_, output_op] : current_op->output_operators) {
    // Get next op's input operands corresponding to current op's output
    const auto& next_input_operands = output_op->input_operands;
    const auto& next_input_op_iter = next_input_operands.find(current_op->name);
    if (next_input_op_iter != next_input_operands.end()) {
      // Get input data spaces for those operands
      std::vector<stensor<T>>& next_input_datas = next_input_op_iter->second->datas;
      // Copy current op output data to next op input data
      for (uint32_t i = 0; i < next_input_datas.size(); ++i) {
        const stensor<T>& layer_output_data = layer_output_datas.at(i);
        if (next_input_datas.at(i) != nullptr) {
          CHECK(next_input_datas.at(i)->shapes() == layer_output_data->shapes());
        }
        next_input_datas.at(i) = layer_output_data;
      }
    }
  }
}

/**
 * @brief 对计算图进行反向拓扑排序，确定算子执行顺序
 *
 * 流程：
 * 1. 从输出节点出发，递归 DFS 标记所有节点的 start_time（后序遍历的逆序）
 * 2. 按 start_time 降序排列算子，得到正向执行顺序
 * 3. 计算每个算子的 end_time（即其最后一个后继的 start_time）
 *    - end_time 用于输出内存复用的生命周期管理
 */
void RuntimeGraph::ReverseTopoSort() {
  // 构建拓扑顺序
  for (const auto& op : operators_) {
    // 根据输入节点构建拓扑排序
    if (op != nullptr && !op->has_forward) {
      int32_t current_forward_idx = 0;
      this->ReverseTopoSortInternal(op, current_forward_idx);
    }
  }

  // 根据拓扑顺序调整算子的执行顺序
  std::sort(operators_.begin(), operators_.end(), [](const auto& op1, const auto& op2) {
    return op1->start_time > op2->start_time;
  });

  int32_t forward_index = 1;
  for (const auto& op : operators_) {
    op->start_time = forward_index;
    forward_index += 1;
  }

  // 计算每个算子的 end_time（其输出数据被最后使用的时刻）
  for (const auto& op : operators_) {
    const auto& next_ops = op->output_operators;
    int32_t last_forward_index = -1;
    for (const auto& [_, next_op] : next_ops) {
      if (next_op->start_time >= last_forward_index) {
        last_forward_index = next_op->start_time;
      }
    }

    if (last_forward_index == -1) {
      op->end_time = op->start_time + 1;
    } else {
      op->end_time = last_forward_index;
    }
    op->occur_end_time = -1;
  }
}

/**
 * @brief 反向拓扑排序的递归内部实现（从输出端向输入端 DFS）
 * @param root_op 当前递归的根算子
 * @param current_forward_idx 当前分配的序号（引用传递，递归递增）
 *
 * 采用后序遍历的方式：先递归处理所有后继节点，再为当前节点分配 start_time。
 * 这样可以确保输入节点获得较大的序号，输出节点获得较小的序号。
 */
template <typename T>
void RuntimeGraph::ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op,
                                           int32_t& current_forward_idx) {
  if (!root_op) {
    LOG(INFO) << "Current operator is nullptr";
    return;
  }
  // 标记输入/输出节点
  if (root_op->input_operands.empty() && !root_op->has_forward) {
    this->input_ops_.push_back(root_op);
  }
  if (root_op->output_names.empty() && !root_op->has_forward) {
    this->output_ops_.push_back(root_op);
  }

  root_op->has_forward = true;
  const auto& next_ops = root_op->output_operators;
  // 先递归访问所有后继节点
  for (const auto& [_, op] : next_ops) {
    if (op != nullptr && !op->has_forward) {
      this->ReverseTopoSortInternal(op, current_forward_idx);
    }
  }

  // 断言所有后继节点都已被访问
  for (const auto& [_, op] : next_ops) {
    CHECK_EQ(op->has_forward, true);
  }
  // 为当前节点分配序号（后序：当前节点序号大于所有后继）
  root_op->start_time = current_forward_idx;
  current_forward_idx += 1;
}

/**
 * @brief 构建算子之间的连接关系，并为每个算子创建对应的 Layer 对象
 *
 * 流程：
 * 1. 遍历每个算子的 output_names，在 operators_ 中查找对应的后继算子，建立映射
 * 2. 对于非输入/输出占位算子，通过 LayerRegisterer 创建对应的 Layer 对象
 * 3. 将 Layer 与 RuntimeOperator 双向绑定
 */
void RuntimeGraph::CreateNodeRelation() {
  // 构建图关系
  for (const auto& current_op : this->operators_) {
    // 获取当前节点的所有后继节点的names，遍历根据next_op_name从operators_maps_中插入所需要的节点
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& kOutputName : output_names) {
      for (const auto& output_op : this->operators_) {
        if (output_op != current_op && output_op->name == kOutputName) {
          current_op->output_operators.insert({kOutputName, output_op});
        }
      }
    }
    // 除了输入和输出节点，都创建layer
    if (current_op->type != "pnnx.Input" && current_op->type != "pnnx.Output") {
      auto layer = RuntimeGraph::CreateLayer(current_op);
      if (layer) {
        current_op->layer = layer;
        layer->set_runtime_operator(current_op);
      } else {
        LOG(FATAL) << "Layer " << current_op->name << " create failed!";
      }
    }
  }
}

RuntimeGraph::GraphState RuntimeGraph::graph_state() const { return this->graph_state_; }

/**
 * @brief 设置指定输入算子的输入数据
 * @param input_name 输入算子的名称
 * @param inputs 输入张量列表
 *
 * 内部通过 PropagateLayerOutputs 将输入数据传播给后继算子。
 */
void RuntimeGraph::set_inputs(const std::string& input_name, const std::vector<sftensor>& inputs) {
  CHECK(this->graph_state_ == GraphState::Complete);
  std::shared_ptr<RuntimeOperator> input_op;
  for (auto op : this->input_ops_) {
    if (op->name == input_name) {
      input_op = op;
      break;
    }
  }
  CHECK(input_op != nullptr) << "Can not find the input operator: " << input_name;
  PropagateLayerOutputs(input_op, inputs);
}

/**
 * @brief 获取指定输出算子的输出数据
 * @param output_name 输出算子的名称
 * @return 输出张量列表
 *
 * 输出算子的数据存储在其 input_operands_seq 中（因为输出算子是 pnnx.Output 占位节点）。
 */
std::vector<sftensor> RuntimeGraph::get_outputs(const std::string& output_name) const {
  CHECK(this->graph_state_ == GraphState::Complete);
  std::shared_ptr<RuntimeOperator> output_op;
  for (auto op : this->output_ops_) {
    if (op->name == output_name) {
      output_op = op;
    }
  }

  CHECK(output_op != nullptr) << "Can not find the output operator: " << output_name;
  std::vector<sftensor> outputs;
  for (const auto& input_operand : output_op->input_operands_seq) {
    std::copy(input_operand->datas.begin(), input_operand->datas.end(),
              std::back_inserter(outputs));
  }
  return outputs;
}

bool RuntimeGraph::is_input_op(const std::string& op_name) const {
  for (auto op : this->input_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

bool RuntimeGraph::is_output_op(const std::string& op_name) const {
  for (auto op : this->output_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

}  // namespace net_infer
