import os
import sys
import numpy as np
import pyxrt
import struct

# Configuration
XCLBIN_PATH = "/home/miahafiz/NeuroRing/krnl_test_kernel_hw.xclbin"  # Update if needed
DEVICE_INDEX = 0
NUM_WORDS = 128  # Must be a multiple of 16

# HBM mapping from TestKernel.cfg
HBM_IN_0 = 20
HBM_OUT_0 = 21
HBM_IN_1 = 0
HBM_OUT_1 = 1


def generate_dummy_input(num_words):
    data = np.zeros(num_words, dtype=np.uint32)
    for i in range(0, num_words, 2):
        dst = (i // 2) & 0xFFFFFF
        delay = (i // 2) & 0xFF
        weight = float(i) * 0.01
        data[i] = (dst << 8) | delay
        data[i+1] = struct.unpack('<I', struct.pack('<f', float(weight)))[0]

    return data

def verify_output(input_data, output_data):
    # For this test, we expect the data to be round-tripped through both kernels
    if np.array_equal(input_data[:NUM_WORDS], output_data[:NUM_WORDS]):
        print("TEST PASSED")
        return True
    else:
        print("TEST FAILED")
        print("Input:", input_data[:32])
        print("Output:", output_data[:32])
        return False

def main():
    device = pyxrt.device(DEVICE_INDEX)
    xclbin = pyxrt.xclbin(XCLBIN_PATH)
    uuid = device.load_xclbin(xclbin)
    # Get kernel handles
    kernel_0 = pyxrt.kernel(device, uuid, "test_kernel:{test_kernel_0}")
    kernel_1 = pyxrt.kernel(device, uuid, "test_kernel:{test_kernel_1}")


    # Allocate input/output buffers for both kernels
    input_data_0 = np.ones(NUM_WORDS, dtype=np.uint32) * 28
    print(f"input_data_0: {input_data_0}")
    input_data_1 = generate_dummy_input(NUM_WORDS)
    print(f"input_data_1: {input_data_1}")
    output_data_1 = np.zeros(NUM_WORDS, dtype=np.uint32)
    output_data_0 = np.zeros(NUM_WORDS, dtype=np.uint32)

    bo_in_0 = pyxrt.bo(device, len(input_data_0)*4, pyxrt.bo.normal, kernel_0.group_id(0))
    bo_out_0 = pyxrt.bo(device, len(input_data_0)*4, pyxrt.bo.normal, kernel_0.group_id(1))
    bo_in_1 = pyxrt.bo(device, len(input_data_0)*4, pyxrt.bo.normal, kernel_1.group_id(0))
    bo_out_1 = pyxrt.bo(device, len(output_data_1)*4, pyxrt.bo.normal, kernel_1.group_id(1))

    print(kernel_0.group_id(0))
    print(kernel_0.group_id(1))
    print(kernel_1.group_id(0))
    print(kernel_1.group_id(1))
    
    bo_in_0.write(input_data_0, 0)
    bo_in_0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in_1.write(input_data_1, 0)
    bo_in_1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    
    print("done writing")

    # Launch both kernels
    # Streams are connected in hardware, so pass None for stream args
    run_0 = kernel_0(bo_in_0, bo_out_0, NUM_WORDS, None, None)
    print("done running 0")
    run_1 = kernel_1(bo_in_1, bo_out_1, NUM_WORDS, None, None)
    print("done running 1")

    # Wait for both kernels to finish
    run_0.wait()
    print("done waiting 0")
    run_1.wait()
    print("done waiting 1")
    
    bo_out_0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output_data_0 = np.frombuffer(bo_out_0.read(len(output_data_0) * 4, 0), dtype=np.uint32)
    print(f"output_data_0: {output_data_0}")

    # Read back output from test_kernel_1
    bo_out_1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output_data_1 = np.frombuffer(bo_out_1.read(len(output_data_1) * 4, 0), dtype=np.uint32)
    print(f"output_data_1: {output_data_1}")

    # Verify output (input to kernel_0 should propagate through both kernels)
    verify_output(input_data_0, output_data_0)

if __name__ == "__main__":
    main() 
