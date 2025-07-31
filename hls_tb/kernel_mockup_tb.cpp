#include "../hls/NeuroRing.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define TEST_NUM_WORDS 32 // Must be a multiple of 16

int main() {
    // Prepare input buffer (hbm_in) with a known pattern
    std::vector<uint32_t> hbm_in(TEST_NUM_WORDS);
    for (uint32_t i = 0; i < TEST_NUM_WORDS; ++i) {
        if (i % 2 == 0) {
            // Even index: DstID (upper 24 bits), Delay (lower 8 bits)
            hbm_in[i] = ((0xABCD00 + i) << 8) | (i & 0xFF);
        } else {
            // Odd index: Weight as float, stored as uint32_t
            float weight = 1.5f + 0.1f * i;
            uint32_t u;
            std::memcpy(&u, &weight, sizeof(float));
            hbm_in[i] = u;
        }
    }

    // Output buffer (hbm_out)
    std::vector<uint32_t> hbm_out(TEST_NUM_WORDS, 0);
    // Output buffer for float weights (hbm_out_float) - 8 floats per 16 words
    std::vector<float> hbm_out_float(TEST_NUM_WORDS / 2, 0.0f);

    // Call kernel (now with float output buffer)
    kernel_mockup(hbm_in.data(), hbm_out.data(), hbm_out_float.data(), TEST_NUM_WORDS);

    // Print input buffer
    std::cout << "Input buffer (hbm_in):\n";
    for (uint32_t i = 0; i < TEST_NUM_WORDS; ++i) {
        std::cout << hbm_in[i] << " ";
    }
    std::cout << "\n";

    // Print output buffer
    std::cout << "Output buffer (hbm_out):\n";
    for (uint32_t i = 0; i < TEST_NUM_WORDS; ++i) {
        std::cout << hbm_out[i] << " ";
    }
    std::cout << "\n";

    // Print float output buffer
    std::cout << "Output buffer (hbm_out_float):\n";
    for (uint32_t i = 0; i < TEST_NUM_WORDS / 2; ++i) {
        std::cout << hbm_out_float[i] << " ";
    }
    std::cout << "\n";

    // Verify the kernel behavior
    std::cout << "\nAnalyzing kernel behavior:\n";
    bool pass = true;
    
    // Check if the kernel is actually transforming data (not just copying)
    bool data_transformed = false;
    for (uint32_t i = 0; i < TEST_NUM_WORDS; i++) {
        if (hbm_in[i] != hbm_out[i]) {
            data_transformed = true;
            break;
        }
    }
    
    if (data_transformed) {
        std::cout << "✓ Kernel is transforming data (as expected)\n";
        
        // Check if weights are being processed (they should not all be 0)
        bool weights_processed = false;
        for (uint32_t i = 1; i < TEST_NUM_WORDS; i += 2) {
            if (hbm_out[i] != 0) {
                weights_processed = true;
                break;
            }
        }
        
        if (weights_processed) {
            std::cout << "✓ Weights are being processed correctly\n";
        } else {
            std::cout << "⚠ Weights appear to be zero - this may indicate a bug in weight handling\n";
            pass = false;
        }
        
        // Check if DstID/Delay pairs are being processed
        bool dst_delay_processed = false;
        for (uint32_t i = 0; i < TEST_NUM_WORDS; i += 2) {
            if (hbm_out[i] != 0) {
                dst_delay_processed = true;
                break;
            }
        }
        
        if (dst_delay_processed) {
            std::cout << "✓ DstID/Delay pairs are being processed\n";
        } else {
            std::cout << "⚠ DstID/Delay pairs appear to be zero - this may indicate a bug\n";
            pass = false;
        }
        
    } else {
        std::cout << "⚠ Kernel appears to be copying data without transformation\n";
        pass = false;
    }
    if (pass) {
        std::cout << "Testbench PASSED!\n";
    } else {
        std::cout << "Testbench FAILED!\n";
    }
    return 0;
}
