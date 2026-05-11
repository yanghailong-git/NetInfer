#include "utils/time/time_logging.hpp"
#include <utility>
#include "layer/abstract/layer_factory.hpp"

namespace net_infer {
namespace utils {

/**
 * @brief 获取 LayerTimeStatesCollector 的单例实例
 * @return 指向 LayerTimeStatesCollector 的共享指针
 *
 * 使用双重检查锁定（DCL）模式，通过 std::lock_guard 保证线程安全。
 * 写的是 DCL，但这段代码**并不是真正的 DCL**，而是**简单的互斥锁保护**：
 */
PtrLayerTimeStatesCollector LayerTimeStatesSingleton::SingletonInstance() {
  std::lock_guard<std::mutex> lock_(mutex_);
  if (time_states_collector_ == nullptr) {
    time_states_collector_ = std::make_shared<LayerTimeStatesCollector>();
  }
  return time_states_collector_;
}

/**
 * @brief 初始化/重置时间统计收集器
 *
 * 若已有旧的收集器实例，先将其释放，再重新创建新的单例实例。
 * 通常在每次 Forward(true) 调用前执行，以清除上一次的计时数据。
 */
void LayerTimeStatesSingleton::LayerTimeStatesCollectorInit() {
  if (time_states_collector_ != nullptr) {
    std::lock_guard<std::mutex> lock_(mutex_);
    time_states_collector_.reset();
    time_states_collector_ = nullptr;
  }
  time_states_collector_ = LayerTimeStatesSingleton::SingletonInstance();
}

std::mutex LayerTimeStatesSingleton::mutex_;

PtrLayerTimeStatesCollector LayerTimeStatesSingleton::time_states_collector_;

/**
 * @brief 构造函数，开始记录当前层的执行时间
 * @param layer_name 层名称
 * @param layer_type 层类型
 *
 * 在单例收集器中插入或初始化该层的计时状态，并记录起始时间点。
 */
LayerTimeLogging::LayerTimeLogging(std::string layer_name, std::string layer_type)
    : layer_name_(std::move(layer_name)),
      layer_type_(std::move(layer_type)),
      start_time_(Time::now()) {
  auto layer_time_states = LayerTimeStatesSingleton::SingletonInstance();
  layer_time_states->insert(
      {layer_name_, std::make_shared<LayerTimeState>(0l, layer_name_, layer_type_)});
}

/**
 * @brief 析构函数，结束记录当前层的执行时间并累加耗时
 *
 * 当 LayerTimeLogging 对象离开作用域时自动调用：
 * 1. 计算从构造到析构的时间差（毫秒）
 * 2. 将耗时累加到对应层的 LayerTimeState::duration_time_ 中
 * 3. 使用 mutex 保证多线程环境下的累加安全
 */
LayerTimeLogging::~LayerTimeLogging() {
  auto layer_time_states = LayerTimeStatesSingleton::SingletonInstance();
  const auto layer_state_iter = layer_time_states->find(layer_name_);
  if (layer_state_iter != layer_time_states->end()) {
    auto& layer_state = layer_state_iter->second;
    CHECK(layer_state != nullptr);

    std::lock_guard<std::mutex> lock_guard(layer_state->time_mutex_);
    const auto end_time = Time::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    layer_state->duration_time_ += duration;
  } else {
    LOG(ERROR) << "Can not find the layer: " << layer_name_ << " in the time logging.";
  }
}

/**
 * @brief 汇总并打印所有层的耗时统计
 *
 * 遍历 LayerTimeStatesCollector 中所有层的计时状态，
 * 打印每个层的名称、类型和累计耗时，最后输出总耗时。
 * 仅打印 duration_time_ 不为 0 的层（避免打印未执行的层）。
 */
void LayerTimeLogging::SummaryLogging() {
  auto layer_time_states = LayerTimeStatesSingleton::SingletonInstance();
  CHECK(layer_time_states != nullptr);
  LayerTimeStatesCollector layer_time_states_collector = *layer_time_states.get();

  long total_time_costs = 0;
  for (const auto& [layer_name, layer_time_state] : layer_time_states_collector) {
    CHECK(layer_time_state != nullptr);

    std::lock_guard<std::mutex> lock(layer_time_state->time_mutex_);
    const auto time_cost = layer_time_state->duration_time_;
    total_time_costs += time_cost;
    if (layer_time_state->duration_time_ != 0) {
      LOG(INFO) << "Layer name: " << layer_name << "\t"
                << "layer type: " << layer_time_state->layer_type_ << "\t"
                << "time cost: " << time_cost << "ms";
    }
  }
  LOG(INFO) << "Total time: " << total_time_costs << "ms";
}

}  // namespace utils
}  // namespace net_infer
