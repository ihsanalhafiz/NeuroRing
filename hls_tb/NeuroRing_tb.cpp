#include "../hls/NeuroRing.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <hls_stream.h>

#define TEST_NEURON_NUM 16
#define TEST_SYNAPSE_LIST_SIZE 32
#define TEST_SIM_TIME 4

int main() {
    // Dummy synapse list: [neuron][synapse entries]
    std::vector<synapse_word_t> SynapseList(TEST_NEURON_NUM * TEST_SYNAPSE_LIST_SIZE, 0);
    // For each neuron, set the first entry as the number of synapses (here, 8 for all)
    for (int i = 0; i < TEST_NEURON_NUM; ++i) {
        SynapseList[i * TEST_SYNAPSE_LIST_SIZE] = 8;
        // Fill 8 dummy synapses
        for (int j = 0; j < 8; ++j) {
            synapse_list_t syn;
            syn.DstID = (i + j) % TEST_NEURON_NUM;
            syn.Delay = 0;
            syn.Weight = 1.0f;
            SynapseList[i * TEST_SYNAPSE_LIST_SIZE + 1 + j] = syn.word;
        }
    }

    // Output spike recorder
    std::vector<uint32_t> SpikeRecorder(TEST_NEURON_NUM * TEST_SIM_TIME, 0);

    // Parameters
    uint32_t SimulationTime = TEST_SIM_TIME;
    float threshold = 0.5f;
    uint32_t AmountOfCores = 1;
    uint32_t NeuronStart = 0;
    uint32_t NeuronTotal = TEST_NEURON_NUM;
    uint32_t DCstimStart = 0;
    uint32_t DCstimTotal = TEST_SIM_TIME;
    float DCstimAmp = 1.0f;
    uint32_t DCneuronStart = 0;
    uint32_t DCneuronTotal = TEST_NEURON_NUM;

    // Call kernel
    hls::stream<synapse_word_t> syn_route_in;
    hls::stream<synapse_word_t> syn_forward_rt;
    NeuroRing(
        SynapseList.data(),
        SimulationTime,
        threshold,
        AmountOfCores,
        NeuronStart,
        NeuronTotal,
        DCstimStart,
        DCstimTotal,
        DCstimAmp,
        DCneuronStart,
        DCneuronTotal,
        SpikeRecorder.data(),
        syn_route_in,
        syn_route_in
    );

    // Print output (SpikeRecorder)
    std::cout << "SpikeRecorder output (first 32 values):\n";
    for (size_t i = 0; i < std::min<size_t>(SpikeRecorder.size(), 32); ++i) {
        std::cout << SpikeRecorder[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Testbench completed." << std::endl;
    return 0;
} 