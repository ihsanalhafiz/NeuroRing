#include <iostream>
#include <cstring>
#include <cstdlib>
#include "../hls/NeuroRing.h"

#define TEST_NEURON_NUM 64
#define TEST_SYNAPSE_LIST_SIZE 100

int main() {
    // Test parameters
    const uint32_t NeuronTotal = TEST_NEURON_NUM;
    const uint32_t SynapseListSize = TEST_SYNAPSE_LIST_SIZE;
    const uint32_t AmountOfCores = 1;
    const uint32_t NeuronStart = 1;
    const uint32_t DCstimStart = 0;
    const uint32_t DCstimTotal = 0;
    const float DCstimAmp = 0.1f;
    const float threshold = 1.0f;
    const uint32_t SimulationTime = 1;

    std::cout << "NeuronTotal: " << NeuronTotal << std::endl;
    std::cout << "SynapseListSize: " << SynapseListSize << std::endl;
    std::cout << "AmountOfCores: " << AmountOfCores << std::endl;
    std::cout << "NeuronStart: " << NeuronStart << std::endl;
    std::cout << "DCstimStart: " << DCstimStart << std::endl;
    std::cout << "DCstimTotal: " << DCstimTotal << std::endl;
    std::cout << "DCstimAmp: " << DCstimAmp << std::endl;

    // Allocate and initialize SynapseList
    uint32_t *SynapseList = new uint32_t[NeuronTotal * SynapseListSize];
    for (uint32_t i = 0; i < NeuronTotal; ++i) {
        // Each neuron has 8 synapses for this test
        SynapseList[i * SynapseListSize] = 8; // first word: number of synapses
        for (uint32_t j = 1; j <= 8; ++j) {
            // Dummy synapse data: dst, delay, weight packed as uint32_t
            int idx = i * SynapseListSize + j;
            if (idx%2 == 0) {
                float_to_uint32 weight_conv;
                weight_conv.f = 1.1f * j;
                SynapseList[idx] = weight_conv.u;
            } else {
                SynapseList[idx] = ((i+j) << 8) & 0xFFFFFF00;
            }
        }
        for (uint32_t j = 9; j < SynapseListSize; ++j) {
            SynapseList[i * SynapseListSize + j] = 0;
        }
    }

    std::cout << "SynapseList done writing" << std::endl;

    // Allocate and initialize SpikeRecorder (input: initial spikes, output: new spikes)
    uint32_t *SpikeRecorder = new uint32_t[NeuronTotal / 32];
    for (uint32_t i = 0; i < NeuronTotal / 32; ++i) {
        SpikeRecorder[i] = 0x00000006; // All neurons spike initially
    }

    // print start of simulation
    std::cout << "Start of simulation" << std::endl;

    // Call the kernel
    NeuroRing_singlestep(
        SynapseList,
        SpikeRecorder,
        SimulationTime,
        threshold,
        AmountOfCores,
        NeuronStart,
        NeuronTotal,
        DCstimStart,
        DCstimTotal,
        DCstimAmp
    );

    // Print output SpikeRecorder
    std::cout << "SpikeRecorder output:" << std::endl;
    for (uint32_t i = 0; i < NeuronTotal / 32; ++i) {
        std::cout << "Word " << i << ": 0x" << std::hex << SpikeRecorder[i] << std::dec << std::endl;
    }

    delete[] SynapseList;
    delete[] SpikeRecorder;
    return 0;
} 