#include "expression.hpp"
#include <stack>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace net_infer {

// 构造函数：初始化表达式层，创建表达式解析器
ExpressionLayer::ExpressionLayer(std::string statement)
    : NonParamLayer("Expression"), statement_(std::move(statement)) {
  parser_ = std::make_unique<ExpressionParser>(statement_);
}

// 判断token是否为操作符（加法或乘法）
bool ExpressionLayer::TokenIsOperator(Token token) const {
  return token.token_type == TokenType::TokenAdd || token.token_type == TokenType::TokenMul;
}

// 前向传播：解析表达式并执行对应的张量运算
// 使用逆波兰表达式（后缀表达式）的栈求值方式处理token序列
StatusCode ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the expression layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the expression layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  CHECK(this->parser_ != nullptr) << "The parser in the expression layer is null!";
  // 对表达式进行分词
  this->parser_->Tokenizer(false);
  const auto& tokens = this->parser_->tokens();
  const auto& token_str_array = this->parser_->token_str_array();
  CHECK(!tokens.empty() && !token_str_array.empty())
      << "The expression parser failed to parse " << statement_;

  const uint32_t batch_size = outputs.size();
  // 使用栈来保存中间运算结果，从右向左遍历token（逆序处理）
  std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
  for (auto iter = tokens.rbegin(); iter != tokens.rend(); ++iter) {
    const auto& current_token = *iter;
    // 如果是数据类型，就将对应分支的input插入到栈中
    if (current_token.token_type == TokenType::TokenInputNumber) {
      // 获取输入分支编号（如@0表示第0个输入分支）
      std::string str_number = *(token_str_array.rbegin() + std::distance(tokens.rbegin(), iter));
      str_number.erase(str_number.begin());

      int32_t input_branch = std::stoi(str_number);
      CHECK(input_branch >= 0) << "Input branch must be >= 0";
      uint32_t input_start_pos = input_branch * batch_size;
      std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
      // 将该分支下的所有batch张量压入栈中
      for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(i + input_start_pos < inputs.size())
            << "The " << i << "th operand doesn't have appropriate number of tensors";
        input_token_nodes.push_back(inputs.at(i + input_start_pos));
      }
      op_stack.push(input_token_nodes);
    } else if (TokenIsOperator(current_token)) {
      // process operation
      // 操作符需要弹出两个操作数进行运算
      CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
      std::shared_ptr<Tensor<float>> (*function)(const std::shared_ptr<Tensor<float>>& tensor1,
                                                 const std::shared_ptr<Tensor<float>>& tensor2);
      std::vector<std::shared_ptr<Tensor<float>>> input_node1(std::move(op_stack.top()));
      CHECK(input_node1.size() == batch_size)
          << "The first operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();
      std::vector<std::shared_ptr<Tensor<float>>> input_node2(std::move(op_stack.top()));
      CHECK(input_node2.size() == batch_size)
          << "The second operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();

      // 根据操作符类型选择对应的张量运算函数
      std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
      if (current_token.token_type == TokenType::TokenAdd) {
        function = TensorElementAdd;
      } else if (current_token.token_type == TokenType::TokenMul) {
        function = TensorElementMultiply;
      } else {
        LOG(FATAL) << "Unsupported operator type in the expression layer: "
                   << int(current_token.token_type);
      }
      // 并行执行batch内每个样本的逐元素运算
#pragma omp parallel for num_threads(batch_size)
      for (uint32_t i = 0; i < batch_size; ++i) {
        output_token_nodes.at(i) = function(input_node1.at(i), input_node2.at(i));
      }
      op_stack.push(output_token_nodes);
    }
  }
  // 最终栈中应只剩下一个结果
  CHECK(op_stack.size() == 1) << "The expression has more than one output operand!";
  std::vector<sftensor> output_node = op_stack.top();
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (outputs.at(i) != nullptr && !outputs.at(i)->empty()) {
      CHECK(outputs.at(i)->shapes() == output_node.at(i)->shapes());
    }
    outputs.at(i) = output_node.at(i);
  }
  return StatusCode::kSuccess;
}

// 从RuntimeOperator创建ExpressionLayer实例
StatusCode ExpressionLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                           std::shared_ptr<Layer<float>>& expression_layer) {
  if (!op) {
    LOG(ERROR) << "The expression operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the expression layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (params.find("expr") == params.end()) {
    return StatusCode::kParseParamError;
  }

  // 获取表达式字符串参数
  auto statement_param = std::dynamic_pointer_cast<RuntimeParameterString>(params.at("expr"));
  if (statement_param == nullptr) {
    LOG(ERROR) << "Can not find the expression parameter";
    return StatusCode::kParseParamError;
  }
  if (statement_param->type != RuntimeParameterType::kParameterString) {
    LOG(ERROR) << "Can not find the expression parameter";
    return StatusCode::kParseParamError;
  }

  expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);
  return StatusCode::kSuccess;
}

// 注册表达式层（对应pnnx.Expression）
LayerRegistererWrapper kExpressionCreateInstance(ExpressionLayer::CreateInstance,
                                                 "pnnx.Expression");
}  // namespace net_infer
