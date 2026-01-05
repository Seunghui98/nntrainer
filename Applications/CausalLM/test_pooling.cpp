#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <cassert>

#include <embedding_pooling_layer.h>
#include <embedding_normalize_layer.h>
#include <tensor.h>
#include <var_grad.h>
#include <layer_context.h>
#include <app_context.h>

using namespace causallm;
using namespace nntrainer;

void test_mean_pooling() {
    std::cout << "[TEST] Mean Pooling" << std::endl;

    // 1. Setup Layer
    EmbeddingPoolingLayer layer;
    std::vector<std::string> props = {
        "word_embedding_dimension=4",
        "pooling_mode_mean_tokens=true"
    };
    layer.setProperty(props);

    // 2. Prepare Input
    // Batch=1, Seq=3, Dim=4
    // Token 0: [1, 1, 1, 1]
    // Token 1: [2, 2, 2, 2]
    // Token 2: [3, 3, 3, 3]
    Tensor input(1, 1, 3, 4);
    float* data = input.getData<float>();
    for (int i=0; i<3; ++i) {
        for (int j=0; j<4; ++j) {
            data[i*4 + j] = (float)(i + 1);
        }
    }

    // 3. Prepare Context
    Tensor output(1, 1, 1, 4);

    Var_Grad input_vg(&input, nullptr);
    Var_Grad output_vg(&output, nullptr);
    
    std::vector<Var_Grad*> inputs = {&input_vg};
    std::vector<Var_Grad*> outputs = {&output_vg};
    std::vector<Var_Grad*> tensors = {};
    std::vector<Weight*> weights = {};

    RunLayerContext context("test_pooling", false, 0.0f, false, 1.0f, nullptr, false,
                            weights, inputs, outputs, tensors);
    
    // Run incremental forwarding for the whole sequence
    // from=0, to=3
    layer.incremental_forwarding(context, 0, 3, false);
    
    // 4. Verify Output
    // Expected: Mean of [1,1,1,1], [2,2,2,2], [3,3,3,3] is [2,2,2,2]
    float* out_data = output.getData<float>();
    bool passed = true;
    for (int j=0; j<4; ++j) {
        if (std::abs(out_data[j] - 2.0f) > 1e-5) {
            passed = false;
            std::cout << "Mismatch at index " << j << ": " << out_data[j] << " != 2.0" << std::endl;
        }
    }
    
    if (passed) std::cout << "-> PASSED" << std::endl;
    else std::cout << "-> FAILED" << std::endl;
}

void test_normalization() {
    std::cout << "[TEST] Normalization" << std::endl;
    
    EmbeddingNormalizeLayer layer;
    
    // Input: [1, 2, 3, 4] -> L2 Norm = sqrt(1+4+9+16) = sqrt(30) ~= 5.477
    Tensor input(1, 1, 1, 4);
    float* data = input.getData<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;
    
    Tensor output(1, 1, 1, 4);
    
    Var_Grad input_vg(&input, nullptr);
    Var_Grad output_vg(&output, nullptr);
    
    std::vector<Var_Grad*> inputs = {&input_vg};
    std::vector<Var_Grad*> outputs = {&output_vg};
    std::vector<Var_Grad*> tensors = {};
    std::vector<Weight*> weights = {};
    
    RunLayerContext context("test_norm", false, 0.0f, false, 1.0f, nullptr, false, 
                            weights, inputs, outputs, tensors);
    
    layer.forwarding(context, false);
    
    float* out = output.getData<float>();
    float sq_sum = 0.0f;
    for(int i=0; i<4; ++i) sq_sum += out[i]*out[i];
    float norm = std::sqrt(sq_sum);
    
    if (std::abs(norm - 1.0f) < 1e-5) {
        std::cout << "-> PASSED (Norm is 1.0)" << std::endl;
    } else {
        std::cout << "-> FAILED (Norm is " << norm << ")" << std::endl;
    }
    
    std::cout << "Values: ";
    for(int i=0; i<4; ++i) std::cout << out[i] << " ";
    std::cout << std::endl;
}

int main() {
    try {
        test_mean_pooling();
        test_normalization();
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
