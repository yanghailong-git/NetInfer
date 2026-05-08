#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace net_infer {

// 初始化全局注册表指针为 nullptr，采用懒加载（Lazy Initialization）策略
LayerRegisterer::CreateRegistry* LayerRegisterer::registry_ = nullptr;

/**
 * @brief 向全局注册表中注册指定类型的层创建器
 * @param layer_type 层类型名称，如 "Convolution", "ReLU" 等
 * @param creator 对应的层创建函数
 *
 * 若该 layer_type 已被注册，将触发 FATAL 日志，防止重复注册。
 */
void LayerRegisterer::RegisterCreator(const std::string& layer_type, const Creator& creator) {
  CHECK(!layer_type.empty());
  CHECK(creator != nullptr);
  CreateRegistry* registry = Registry();
  CHECK_EQ(registry->count(layer_type), 0)
      << "Layer type: " << layer_type << " has already registered!";
  registry->insert({layer_type, creator});
}

/**
 * @brief 获取全局层注册表的单例指针
 * @return 指向 CreateRegistry 的指针
 *
 * 首次调用时会 new 出注册表实例，并通过 RegistryGarbageCollector 在程序结束时自动回收。
 */
LayerRegisterer::CreateRegistry* LayerRegisterer::Registry() {
  if (registry_ == nullptr) {
    registry_ = new CreateRegistry();
    static RegistryGarbageCollector c;
  }

  CHECK(registry_ != nullptr) << "Global layer register init failed!";
  return registry_;
}

/**
 * @brief 根据 RuntimeOperator 创建对应的层对象
 * @param op 运行时算子，包含层的类型和参数配置
 * @return 创建完成的层对象（float 类型）
 *
 * 流程：
 * 1. 从 Registry 中查找对应 layer_type 的创建器
 * 2. 调用创建器生成层对象
 * 3. 检查创建状态，失败则触发 FATAL 日志
 */
std::shared_ptr<Layer<float>> LayerRegisterer::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
  CreateRegistry* registry = Registry();
  const std::string& layer_type = op->type;
  LOG_IF(FATAL, registry->count(layer_type) <= 0) << "Can not find the layer type: " << layer_type;
  const auto& creator = registry->find(layer_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer<float>> layer;
  const auto& status = creator(op, layer);
  LOG_IF(FATAL, status != StatusCode::kSuccess)
      << "Create the layer: " << layer_type << " failed, error code: " << int32_t(status);
  return layer;
}

/**
 * @brief 获取当前已注册的所有层类型名称列表
 * @return 按字母序排列的层类型名称向量
 */
std::vector<std::string> LayerRegisterer::layer_types() {
  std::set<std::string> layer_types_unique;
  CreateRegistry* registry = Registry();
  for (const auto& [layer_type, _] : *registry) {
    layer_types_unique.insert(layer_type);
  }
  std::vector<std::string> layer_types(layer_types_unique.begin(), layer_types_unique.end());
  return layer_types;
}

}  // namespace net_infer
