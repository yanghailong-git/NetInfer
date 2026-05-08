#include "data/load_data.hpp"
#include <glog/logging.h>
#include <armadillo>
#include <fstream>
#include <string>
#include <utility>

namespace net_infer {

/**
 * @brief 计算 CSV 文件对应的矩阵大小（行数和最大列数）
 * @param file 输入文件流，函数执行后文件指针会恢复到原始位置
 * @param split_char 分隔符，例如 ',' 或 '\t'
 * @return 返回一个 pair，first 为行数，second 为最大列数
 *
 * 该函数通过逐行读取文件内容，统计总行数以及每一行的列数，
 * 最终返回最大列数，以便后续加载时预先分配矩阵空间。
 */
std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream& file,
                                                        char split_char) {
  // 保存文件当前的 good 状态，并在后续恢复
  bool load_ok = file.good();
  file.clear();

  size_t fn_rows = 0;  // 统计文件总行数
  size_t fn_cols = 0;  // 统计最大列数
  const std::ifstream::pos_type start_pos = file.tellg();

  std::string token;
  std::string line_str;
  std::stringstream line_stream;

  // 逐行读取文件，统计行列数
  while (file.good() && load_ok) {
    std::getline(file, line_str);
    if (line_str.empty()) {
      break;
    }

    line_stream.clear();
    line_stream.str(line_str);
    size_t line_cols = 0;

    // 使用分隔符拆分当前行，统计列数
    std::string row_token;
    while (line_stream.good()) {
      std::getline(line_stream, row_token, split_char);
      ++line_cols;
    }
    if (line_cols > fn_cols) {
      fn_cols = line_cols;
    }

    ++fn_rows;
  }

  // 恢复文件流状态，将读取位置重置为函数调用前的位置
  file.clear();
  file.seekg(start_pos);
  return {fn_rows, fn_cols};
}

}  // namespace net_infer
