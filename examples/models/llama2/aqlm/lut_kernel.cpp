#include "lut_kernel.h"

#include <numeric>
#include <functional>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <executorch/kernels/optimized/blas/CPUBlas.h>


template<typename fp_dtype>
void quadruple_for(
    int num_inputs,
    int num_input_groups, const fp_dtype* __restrict__ lut,
    int out_features, const uint8_t* __restrict__ b_alt,
    fp_dtype* __restrict__ output_vec
)
{
    std::memset(output_vec, 0, num_inputs * out_features * sizeof(fp_dtype));

    const int lut_stride = num_input_groups * 2 * 256;
    const int b_alt_stride = 2 * out_features;

    for (int input = 0; input < num_inputs; ++input) {
        for (int j = 0; j < num_input_groups; ++j) {
            const fp_dtype* lut_ptr = lut + input * lut_stride + j * 2 * 256;
            const uint8_t* b_alt_ptr = b_alt + j * b_alt_stride;

            for (int i = 0; i < out_features; ++i) {
                output_vec[input * out_features + i] += lut_ptr[b_alt_ptr[i * 2]];
                output_vec[input * out_features + i] += lut_ptr[256 + b_alt_ptr[i * 2 + 1]];
            }
        }
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
void row_wise_scaling_and_bias(
    float* __restrict__ out,
    const float* __restrict__ scales, const float* __restrict__ bias,
    int num_input_vectors, int out_features
) {
    for (int j = 0; j < out_features; j += 4) {
        float32x4_t scales_vec = vld1q_f32(scales + j);
        float32x4_t bias_vec;
        if (bias != nullptr){
            bias_vec = vld1q_f32(bias + j);
        }
        for (int i=0; i < num_input_vectors; ++i) {
            float32x4_t values_vec = vld1q_f32(out + i * out_features + j);
            values_vec = vmulq_f32(values_vec, scales_vec);
            if (bias != nullptr) {
                values_vec = vaddq_f32(values_vec, bias_vec);
            }
            vst1q_f32(out + i * out_features + j, values_vec);
        }
    }
}
#else
void row_wise_scaling_and_bias(
    float* __restrict__ out,
    const float* __restrict__ scales, const float* __restrict__ bias,
    int num_input_vectors, int out_features
) {
    for (int j = 0; j < out_features; ++j) {
        float scale_value = scales[j];
        float bias_value;
        if (bias != nullptr){
            bias_value = bias[j];
        }
        for (int i=0; i < num_input_vectors; ++i) {
            out[i * out_features + j] *= scale_value;
            if (bias != nullptr) {
                out[i * out_features + j] += bias_value;
            }
        }
    }
}
#endif

namespace torch {
  namespace executor {
    namespace native {
      Tensor& code2x8_lut_matmat_out(
        RuntimeContext& ctx,
        const Tensor& input,
        const Tensor& codes,
        const Tensor& codebooks,
        const Tensor& scales,
        const optional<Tensor>& bias,
        Tensor& out
      ) {
        auto input_sizes = input.sizes();
        auto out_features = codes.size(1) * codebooks.size(2);
        auto input_vector_size = input.size(input.dim() - 1);
        auto num_input_vectors = std::accumulate(input_sizes.begin(), input_sizes.end(), 1, std::multiplies<int64_t>()) / input_vector_size;

        // Allocate LUT
        auto lut_data = ctx.allocate_temp(
            4 * num_input_vectors * input_vector_size / 8 * codebooks.size(0) * codebooks.size(1)
        ).get();

        // A @ B.T
        ::executorch::cpublas::gemm(
            ::executorch::cpublas::TransposeType::Transpose,
            ::executorch::cpublas::TransposeType::NoTranspose,
            (int64_t)codebooks.size(0) * codebooks.size(1),      // B rows
            (int64_t)num_input_vectors * input_vector_size / 8,  // A rows
            (int64_t)8,                                          // MatMul dim size
            1.f,
            (float*)codebooks.const_data_ptr(), (int64_t)8,
            (float*)input.const_data_ptr(), (int64_t)8,
            0.f,
            (float*)lut_data, (int64_t)codebooks.size(0) * codebooks.size(1)
        );

        // Do lookup matmul
        quadruple_for<float>(
            num_input_vectors,
            input_vector_size / 8,
            (const float*)lut_data,
            out_features,
            (const uint8_t*)codes.const_data_ptr(),
            (float*)out.mutable_data_ptr()
        );

        const float* bias_ptr = nullptr;
        if (bias.has_value()) {
            bias_ptr = bias.value().const_data_ptr<float>();
        }

        row_wise_scaling_and_bias(
            out.mutable_data_ptr<float>(),
            scales.const_data_ptr<float>(),
            bias_ptr,
            num_input_vectors,
            out_features
        );
        
        return out;
      }
    } // namespace native
  } // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(aqlm, "code2x8_lut_matmat.out", torch::executor::native::code2x8_lut_matmat_out);
