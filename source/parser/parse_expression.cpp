#include "parser/parse_expression.hpp"
#include <glog/logging.h>
#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>

namespace net_infer {

/**
 * @brief 将表达式语法树进行后序遍历，生成逆波兰式序列
 * @param root_node 表达式树的根节点
 * @param reverse_polish 输出的逆波兰式节点序列
 *
 * 逆波兰式（Reverse Polish Notation, RPN）即后缀表达式，
 * 通过后序遍历二叉树得到，便于后续栈式求值。
 */
void ReversePolish(const std::shared_ptr<TokenNode>& root_node,
                   std::vector<std::shared_ptr<TokenNode>>& reverse_polish) {
  if (root_node != nullptr) {
    ReversePolish(root_node->left, reverse_polish);
    ReversePolish(root_node->right, reverse_polish);
    reverse_polish.push_back(root_node);
  }
}

/**
 * @brief 将表达式字符串进行词法分析，生成 Token 序列
 * @param retokenize 若为 true，则强制重新分词；否则仅在 tokens_ 为空时执行
 *
 * 支持的语法单元：
 * - "add"  -> TokenAdd (加法操作)
 * - "mul"  -> TokenMul (乘法操作)
 * - "@N"   -> TokenInputNumber (第 N 个输入，N 为数字)
 * - ","    -> TokenComma (参数分隔符)
 * - "(" ")"-> TokenLeftBracket / TokenRightBracket (括号)
 *
 * 分词前会先去除所有空白字符。
 */
void ExpressionParser::Tokenizer(bool retokenize) {
  if (!retokenize && !this->tokens_.empty()) {
    return;
  }

  CHECK(!statement_.empty()) << "The input statement is empty!";
  // 去除表达式中的所有空白字符
  statement_.erase(
      std::remove_if(statement_.begin(), statement_.end(), [](char c) { return std::isspace(c); }),
      statement_.end());
  CHECK(!statement_.empty()) << "The input statement is empty!";

  for (int32_t i = 0; i < statement_.size();) {
    char c = statement_.at(i);
    if (c == 'a') {
      // 解析 "add" 操作符
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
          << "Parse add token failed, illegal character: " << statement_.at(i + 1);
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
          << "Parse add token failed, illegal character: " << statement_.at(i + 2);
      Token token(TokenType::TokenAdd, i, i + 3);
      tokens_.push_back(token);
      std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } else if (c == 'm') {
      // 解析 "mul" 操作符
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
          << "Parse multiply token failed, illegal character: " << statement_.at(i + 1);
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
          << "Parse multiply token failed, illegal character: " << statement_.at(i + 2);
      Token token(TokenType::TokenMul, i, i + 3);
      tokens_.push_back(token);
      std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } else if (c == '@') {
      // 解析输入索引标记，如 @0, @1
      CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
          << "Parse number token failed, illegal character: " << statement_.at(i + 1);
      int32_t j = i + 1;
      for (; j < statement_.size(); ++j) {
        if (!std::isdigit(statement_.at(j))) {
          break;
        }
      }
      Token token(TokenType::TokenInputNumber, i, j);
      CHECK(token.start_pos < token.end_pos);
      tokens_.push_back(token);
      std::string token_input_number = std::string(statement_.begin() + i, statement_.begin() + j);
      token_strs_.push_back(token_input_number);
      i = j;
    } else if (c == ',') {
      Token token(TokenType::TokenComma, i, i + 1);
      tokens_.push_back(token);
      std::string token_comma = std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_comma);
      i += 1;
    } else if (c == '(') {
      Token token(TokenType::TokenLeftBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_left_bracket =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_left_bracket);
      i += 1;
    } else if (c == ')') {
      Token token(TokenType::TokenRightBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_right_bracket =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_right_bracket);
      i += 1;
    } else {
      LOG(FATAL) << "Unknown  illegal character: " << c;
    }
  }
}

const std::vector<Token>& ExpressionParser::tokens() const { return this->tokens_; }

const std::vector<std::string>& ExpressionParser::token_str_array() const { return this->token_strs_; }

/**
 * @brief 递归生成表达式语法树（内部实现）
 * @param index 当前处理的 token 索引（引用，递归过程中会递增）
 * @return 当前子树对应的节点
 *
 * 文法规则：
 * - 叶子节点：TokenInputNumber（如 @0, @1）
 * - 内部节点：TokenAdd 或 TokenMul，后跟 (left, right) 形式的两个子节点
 *
 * 例如：add(mul(@0,@1),@2) 会被解析为：
 *        add
 *       /   \
 *     mul    @2
 *    /   \
 *  @0    @1
 */
std::shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t& index) {
  CHECK(index < this->tokens_.size());
  const auto current_token = this->tokens_.at(index);
  CHECK(current_token.token_type == TokenType::TokenInputNumber ||
        current_token.token_type == TokenType::TokenAdd ||
        current_token.token_type == TokenType::TokenMul);
  if (current_token.token_type == TokenType::TokenInputNumber) {
    // 叶子节点：提取 @ 后的数字作为输入索引
    uint32_t start_pos = current_token.start_pos + 1;
    uint32_t end_pos = current_token.end_pos;
    CHECK(end_pos > start_pos || end_pos <= this->statement_.length())
        << "Current token has a wrong length";
    const std::string& str_number =
        std::string(this->statement_.begin() + start_pos, this->statement_.begin() + end_pos);
    return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);

  } else if (current_token.token_type == TokenType::TokenMul ||
             current_token.token_type == TokenType::TokenAdd) {
    // 内部节点：创建操作符节点，然后递归解析左右子树
    std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
    current_node->num_index = int32_t(current_token.token_type);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing left bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
    const auto left_token = this->tokens_.at(index);

    if (left_token.token_type == TokenType::TokenInputNumber ||
        left_token.token_type == TokenType::TokenAdd ||
        left_token.token_type == TokenType::TokenMul) {
      current_node->left = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int32_t(left_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing comma!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond right token!";
    const auto right_token = this->tokens_.at(index);
    if (right_token.token_type == TokenType::TokenInputNumber ||
        right_token.token_type == TokenType::TokenAdd ||
        right_token.token_type == TokenType::TokenMul) {
      current_node->right = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int32_t(right_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing right bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
    return current_node;
  } else {
    LOG(FATAL) << "Unknown token type: " << int32_t(current_token.token_type);
  }
}

/**
 * @brief 表达式解析主入口：分词 -> 生成语法树 -> 转逆波兰式
 * @return 逆波兰式节点序列
 *
 * 流程：
 * 1. 若未分词，调用 Tokenizer 进行词法分析
 * 2. 调用 Generate_ 递归生成语法树
 * 3. 对语法树进行后序遍历，得到逆波兰式
 */
std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate() {
  if (this->tokens_.empty()) {
    this->Tokenizer(true);
  }
  int32_t index = 0;
  std::shared_ptr<TokenNode> root = Generate_(index);
  CHECK(root != nullptr);
  CHECK(index == tokens_.size() - 1);

  // 转逆波兰式,之后转移到expression中
  std::vector<std::shared_ptr<TokenNode>> reverse_polish;
  ReversePolish(root, reverse_polish);

  return reverse_polish;
}

TokenNode::TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
                     std::shared_ptr<TokenNode> right)
    : num_index(num_index), left(left), right(right) {}

}  // namespace net_infer
