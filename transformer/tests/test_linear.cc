#include <cmath>

#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_FPLinear_int4() {
    const int m = 1, n = 32000, k = 4096;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    hidden_states.load("tests/assets/input.bin");
    outputGT.load("tests/assets/output.bin");

    // quantize the weight to int4
    Matrix3D<uint8_t> int4_weight((uint8_t *)mem_buf.get_int8buffer(n * k / 2), 1, n, k / 2);
    // Linear_FP_int4 int4_op;
    Linear_FP_int4 int4_op = Linear_FP_int4(int4_weight, "INT4/models/LLaMA_7B_2_chat/lm_head/");

    Matrix3D<float> outputQ(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_simd(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_fast(mem_buf.get_fpbuffer(m * n), 1, m, n);

    // warm up
    for (int i = 0; i < 1; i++) {
        int4_op.forward(hidden_states, outputQ_fast);
    }

    const int flops = k * m * n * 2;
    int4_op.forward_ref(hidden_states, outputQ);

    for (int i = 0; i < 10; i++) {
        STATS_FLOPS(int4_op.profile_name, flops);
        int4_op.forward(hidden_states, outputQ_fast);
        STATS_END(int4_op.profile_name);
    }
    bool success = check_two_equal(outputQ.m_data, outputQ_fast.m_data, outputQ_fast.length(), 1e-3);

    if (!success) {
        std::cout << "-------- Sanity check of " << int4_op.profile_name << " implementation: Fail! -------- "
                  << std::endl;
        exit(-1);
    } else
        std::cout << "-------- Sanity check of " << int4_op.profile_name << " implementation: Passed! -------- "
                  << std::endl;
}
void simple_test_FPLinear_int4() {
    const int m = 2, n = 32, k = 128;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<uint8_t> int4_weight((uint8_t *)mem_buf.get_int8buffer(n * k / 2), 1, n, k / 2);
    Matrix3D<float> scale(mem_buf.get_fpbuffer(n * k / QK), 1, m, n);
    Matrix3D<float> offset(mem_buf.get_fpbuffer(n * k / QK), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    int hidden_states_size = m * k ;
    int int4_weight_size = n * k * sizeof(uint8_t) / 2; 
    int scale_size = n * k / QK;
    int offset_size = n * k / QK;
    // memset works for 0x22 0x11 this kind of set or zero. for other values we have to use loop.
    for (int i = 0; i < hidden_states_size; i++){
        hidden_states.m_data[i] = i;
    }
    // memset works for 0x22 0x11 this kind of set or zero. for other values we have to use loop.
    memset(int4_weight.m_data, 0xFF, int4_weight_size/2); // 32x64 int4 weight, set the first half to 1 and the second half to 2 
    memset(int4_weight.m_data+int4_weight_size/2, 0xEE, int4_weight_size/2);
    //
    for (int i = 0; i < scale_size; i++) {
        scale.m_data[i] = 1.0f; // set all scales to 0.1
        if (i >= scale_size/2) {
            scale.m_data[i] = 2.0f; // set the second half of scales to 0.1
        }
    }

    memset(offset.m_data, 0, offset_size*sizeof(float)); // set all offsets to 0
    struct matmul_params params;
    params.A.row = hidden_states.m_dim_y;
    params.A.column = hidden_states.m_dim_z;
    params.A.data_ptr = hidden_states.m_data;
    params.B.row = int4_weight.m_dim_y;     // k
    params.B.column = int4_weight.m_dim_z;  // n
    params.B.int4_data_ptr = int4_weight.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.data_ptr = output.m_data;
    params.opt_params.num_thread = 1;
    params.scales = scale.m_data;
    params.offset = offset.m_data;
    params.block_size = QK;
    static int8_t *x_int8 = nullptr;
    static float *x_scale = nullptr;
    matmul::MatmulOperator op = matmul::MatmulOperator();
    allocate_aligned_memory(x_int8, MAX_LINEAR_LENGTH * sizeof(int8_t));
    allocate_aligned_memory(x_scale, (MAX_LINEAR_LENGTH / QK) * sizeof(float));

    params.A.int8_data_ptr = x_int8;
    params.A_scales = x_scale;
    op.mat_mul_loop_unrolling(&params);
    for (int i = 0; i < m * n; i++) {
        std::cout << output.m_data[i] << " ";
    }
}
int main() {
    //simple_test_FPLinear_int4();
    test_FPLinear_int4();
    
    Profiler::getInstance().report_internal();
}
