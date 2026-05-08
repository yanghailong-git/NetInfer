#ifndef NET_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#define NET_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#include "layer/abstract/non_param_layer.hpp"
#include "parser/parse_expression.hpp"

namespace net_infer {

// 表达式层：用于解析和执行由加法、乘法组成的张量表达式
// 典型应用场景是处理PNNX模型中的表达式节点（如y=x+y, y=x*y等）
class ExpressionLayer : public NonParamLayer {
 public:
  // 构造函数：传入表达式字符串（如"add(@0,@1)"）
  explicit ExpressionLayer(std::string statement);

  // 前向传播：解析表达式并对输入张量执行相应运算
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 判断token是否为操作符（加法或乘法）
  bool TokenIsOperator(Token token) const;

  // 从RuntimeOperator创建ExpressionLayer实例的工厂方法
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& expression_layer);

 private:
  // 表达式字符串（如"add(@0,@1)"）
  std::string statement_;
  // 表达式解析器，用于将字符串解析为token序列
  std::unique_ptr<ExpressionParser> parser_;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
