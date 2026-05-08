#include "simd.hpp"
#include <glog/logging.h>
#include <armadillo>
#include "utils/math/fmath.hpp"

namespace net_infer {

namespace activation {

// Computes the sigmoid activation function element-wise using SIMD (AVX2/SSE)
// for the whole tensor, followed by a scalar fallback for any remaining elements.
static void SigmoidSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index = 0;
  int64_t packet_size;
  int64_t in_size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  // AVX2 processes 8 floats per iteration.
  packet_size = 8;
  __m256 one = _mm256_set1_ps(1.f);
  __m256 zero = _mm256_setzero_ps();
  for (; index <= in_size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    p = _mm256_div_ps(one, _mm256_add_ps(one, fmath::exp_ps256(_mm256_sub_ps(zero, p))));
    _mm256_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE2__
  // SSE2 processes 4 floats per iteration for the remaining elements.
  packet_size = 4;
  __m128 one128 = _mm_set1_ps(1.f);
  __m128 zero128 = _mm_setzero_ps();
  for (; index <= in_size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    p = _mm_div_ps(one128, _mm_add_ps(one128, fmath::exp_ps(_mm_sub_ps(zero128, p))));
    _mm_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  // Scalar fallback for the tail elements that do not fill a SIMD packet.
  if (index < in_size) {
    while (index < in_size) {
      float value = input->index(index);
      output->index(index) = 1 / (1.f + fmath::exp(-value));
      index += 1;
    }
  }
}

// Computes the ReLU activation function element-wise using SIMD (AVX2/SSE):
// output = max(0, input).
static void ReluSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_setzero_ps();
  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    __m256 value = _mm256_max_ps(zero, p);
    _mm256_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE__
  packet_size = 4;
  __m128 zero128 = _mm_setzero_ps();
  for (; index <= size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    __m128 value = _mm_max_ps(zero128, p);
    _mm_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  // Scalar fallback for the remaining elements.
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      output->index(index) = std::max(value, 0.f);
      index += 1;
    }
  }
}

// Computes the ReLU6 activation function element-wise using SIMD (AVX2/SSE):
// output = min(max(0, input), 6).
static void Relu6SSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_setzero_ps();
  __m256 six = _mm256_set1_ps(6.f);

  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    __m256 value = _mm256_min_ps(_mm256_max_ps(zero, p), six);
    _mm256_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE__
  packet_size = 4;
  __m128 zero128 = _mm_setzero_ps();
  __m128 six128 = _mm_set1_ps(6.f);
  for (; index <= size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    __m128 value = _mm_min_ps(_mm_max_ps(zero128, p), six128);
    _mm_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  // Scalar fallback for the remaining elements.
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      output->index(index) = std::min(std::max(value, 0.f), 6.f);
      index += 1;
    }
  }
}

// Computes the SiLU (Sigmoid Linear Unit) activation function element-wise using SIMD:
// output = input / (1 + exp(-input)).
static void SiluSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 one_256 = _mm256_set1_ps(1.f);
  __m256 zero_256 = _mm256_setzero_ps();

  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    p = _mm256_div_ps(p, _mm256_add_ps(one_256, fmath::exp_ps256(_mm256_sub_ps(zero_256, p))));
    _mm256_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE__
  packet_size = 4;
  __m128 one128 = _mm_set1_ps(1.f);
  __m128 zero128 = _mm_setzero_ps();

  for (; index <= size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    p = _mm_div_ps(p, _mm_add_ps(one128, fmath::exp_ps(_mm_sub_ps(zero128, p))));
    _mm_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  // Scalar fallback for the remaining elements.
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      output->index(index) = value / (1.f + fmath::exp(-value));
      index += 1;
    }
  }
}

// Computes the HardSwish activation function element-wise using SIMD:
//   0                         if x <= -3
//   x                         if x >= 3
//   x * (x + 3) / 6           otherwise
static void HardSwishSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";

  int64_t index;
  float threshold = 3.f;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());

  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_set1_ps(0.f);
  __m256 three = _mm256_set1_ps(threshold);
  __m256 six = _mm256_set1_ps(6.f);
  __m256 minus_three = _mm256_set1_ps(-threshold);
  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 x = _mm256_loadu_ps(in_ptr);

    // Build masks for the three piece-wise branches.
    __m256 le_branch = _mm256_cmp_ps(x, minus_three, _CMP_LE_OS);  // <= -3
    __m256 ge_branch = _mm256_cmp_ps(x, three, _CMP_GE_OS);        // >= 3
    __m256 mid_branch = _mm256_and_ps(_mm256_cmp_ps(x, minus_three, _CMP_GT_OS),
                                      _mm256_cmp_ps(x, three, _CMP_LT_OS));  // -3 < x < 3

    // Compute each branch and mask out the inactive lanes.
    __m256 f1 = _mm256_and_ps(zero, le_branch);
    __m256 f2 = _mm256_and_ps(x, ge_branch);
    __m256 f3 =
        _mm256_and_ps(_mm256_div_ps(_mm256_mul_ps(x, _mm256_add_ps(x, three)), six), mid_branch);

    __m256 result = _mm256_add_ps(_mm256_add_ps(f1, f2), f3);
    _mm256_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE2__
  packet_size = 4;
  __m128 zero128 = _mm_set1_ps(0.f);
  __m128 three128 = _mm_set1_ps(threshold);
  __m128 six128 = _mm_set1_ps(6.f);
  __m128 minus_three128 = _mm_set1_ps(-threshold);
  for (; index <= size - packet_size; index += packet_size) {
    __m128 x = _mm_loadu_ps(in_ptr);

    __m128 le_branch = _mm_cmple_ps(x, minus_three128);  // <= -3
    __m128 ge_branch = _mm_cmpge_ps(x, three128);        // >= 3
    __m128 mid_branch =
        _mm_and_ps(_mm_cmpgt_ps(x, minus_three128), _mm_cmplt_ps(x, three128));  // -3 < x < 3

    __m128 f1 = _mm_and_ps(zero128, le_branch);
    __m128 f2 = _mm_and_ps(x, ge_branch);
    __m128 f3 = _mm_and_ps(_mm_div_ps(_mm_mul_ps(x, _mm_add_ps(x, three128)), six128), mid_branch);

    __m128 result = _mm_add_ps(_mm_add_ps(f1, f2), f3);
    _mm_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  // Scalar fallback for the tail elements.
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      float result = 0.f;
      if (value <= -3.f) {
        result = 0.f;
      } else if (value >= 3.f) {
        result = value;
      } else {
        result = value * (value + threshold) / 6;
      }
      output->index(index) = result;
      index += 1;
    }
  }
}

// Computes the HardSigmoid activation function element-wise using SIMD:
//   0                         if x <= -3
//   1                         if x >= 3
//   x / 6 + 0.5               otherwise
static void HardSigmoidSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";

  int64_t index;
  float threshold = 3.f;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());

  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_set1_ps(0.f);
  __m256 one = _mm256_set1_ps(1.f);

  __m256 three = _mm256_set1_ps(threshold);
  __m256 six = _mm256_set1_ps(6.f);
  __m256 point_five = _mm256_set1_ps(0.5f);
  __m256 minus_three = _mm256_set1_ps(-threshold);
  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 x = _mm256_loadu_ps(in_ptr);
    __m256 le_branch = _mm256_cmp_ps(x, minus_three, _CMP_LE_OS);  // <= -3
    __m256 ge_branch = _mm256_cmp_ps(x, three, _CMP_GE_OS);        // >= 3
    __m256 mid_branch = _mm256_and_ps(_mm256_cmp_ps(x, minus_three, _CMP_GT_OS),
                                      _mm256_cmp_ps(x, three, _CMP_LT_OS));  // -3 < x < 3

    __m256 f1 = _mm256_and_ps(zero, le_branch);
    __m256 f2 = _mm256_and_ps(one, ge_branch);
    __m256 f3 = _mm256_and_ps(_mm256_add_ps(_mm256_div_ps(x, six), point_five), mid_branch);

    __m256 result = _mm256_add_ps(_mm256_add_ps(f1, f2), f3);
    _mm256_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE2__
  packet_size = 4;
  __m128 zero128 = _mm_set1_ps(0.f);
  __m128 one128 = _mm_set1_ps(1.f);

  __m128 three128 = _mm_set1_ps(threshold);
  __m128 six128 = _mm_set1_ps(6.f);
  __m128 point_five128 = _mm_set1_ps(0.5f);
  __m128 minus_three128 = _mm_set1_ps(-threshold);
  for (; index <= size - packet_size; index += packet_size) {
    __m128 x = _mm_loadu_ps(in_ptr);
    __m128 le_branch = _mm_cmp_ps(x, minus_three128, _CMP_LE_OS);  // <= -3
    __m128 ge_branch = _mm_cmp_ps(x, three128, _CMP_GE_OS);        // >= 3
    __m128 mid_branch = _mm_and_ps(_mm_cmp_ps(x, minus_three128, _CMP_GT_OS),
                                   _mm_cmp_ps(x, three128, _CMP_LT_OS));  // -3 < x < 3

    __m128 f1 = _mm_and_ps(zero128, le_branch);
    __m128 f2 = _mm_and_ps(one128, ge_branch);
    __m128 f3 = _mm_and_ps(_mm_add_ps(_mm_div_ps(x, six128), point_five128), mid_branch);

    __m128 result = _mm_add_ps(_mm_add_ps(f1, f2), f3);
    _mm_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  // Scalar fallback for the tail elements.
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      float result = 0.f;
      if (value <= -3.f) {
        result = 0.f;
      } else if (value >= 3.f) {
        result = 1.f;
      } else {
        result = value / 6.f + 0.5f;
      }
      output->index(index) = result;
      index += 1;
    }
  }
}

// Returns the SIMD-accelerated activation function corresponding to the given type.
ActivationFunc ApplySSEActivation(ActivationType act_type) {
  ActivationFunc function;
  switch (act_type) {
    case ActivationType::kActivationRelu: {
      function = ReluSSE;
      return function;
    }
    case ActivationType::kActivationRelu6: {
      function = Relu6SSE;
      return function;
    }
    case ActivationType::kActivationSigmoid: {
      function = SigmoidSSE;
      return function;
    }
    case ActivationType::kActivationSilu: {
      function = SiluSSE;
      return function;
    }
    case ActivationType::kActivationHardSwish: {
      function = HardSwishSSE;
      return function;
    }
    case ActivationType::kActivationHardSigmoid: {
      function = HardSigmoidSSE;
      return function;
    }
    default: {
      LOG(FATAL) << "Unknown SSE activation type: " << int32_t(act_type);
    }
  }
}
}  // namespace activation
}  // namespace net_infer
