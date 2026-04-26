#ifndef NET_INFER_INCLUDE_COMMON_HPP_
#define NET_INFER_INCLUDE_COMMON_HPP_
namespace net_infer {

enum class RuntimeParameterType {
  kParameterUnknown = 0,
  kParameterBool = 1,
  kParameterInt = 2,

  kParameterFloat = 3,
  kParameterString = 4,
  kParameterIntArray = 5,
  kParameterFloatArray = 6,
  kParameterStringArray = 7,
};

enum class StatusCode {
  kUnknownCode = -1,
  kSuccess = 0,

  kInferInputsEmpty = 1,
  kInferOutputsEmpty = 2,
  kInferParamError = 3,
  kInferDimMismatch = 4,

  kFunctionNotImplement = 5,
  kParseWeightError = 6,
  kParseParamError = 7,
  kParseNullOperator = 8,
};

}  // namespace net_infer
#endif  // NET_INFER_INCLUDE_COMMON_HPP_
