#ifndef NEURORING_H
#define NEURORING_H

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_float.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>

#define NEURON_NUM 2048
#define SYNAPSE_LIST_SIZE 5000

typedef ap_uint<64>        synapse_word_t;
typedef uint32_t       spike_status_t;
typedef ap_uint<24>        DstID_t;
typedef ap_uint<8>         Delay_t;
typedef float     Weight_t;

typedef ap_axiu<2048, 0, 0, 0> stream2048u_t;
typedef ap_axiu<64, 0, 0, 0> stream64u_t;
typedef ap_axiu<512, 0, 0, 0> stream512u_t;

union float_to_uint32 {
    float f;
    uint32_t u;
};

struct StaticParams {
    float        dt;
    float        tau;
};
struct SpikeOut_t {
    hls::vector<spike_status_t, NEURON_NUM/32> data;
};
struct SynapseStream_t {
    hls::vector<synapse_word_t, 8> data;
};

extern "C" void AxonLoader(
    uint32_t                    *SynapseList,
    uint32_t                    *SpikeRecorder,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     DCstimStart,
    uint32_t                     DCstimTotal,
    uint32_t                     DCstimAmp,
    uint32_t                     SimulationTime,
    hls::stream<stream2048u_t>   &SpikeOutIn,
    hls::stream<stream512u_t>    &SynapseStream);

extern "C" void NeuroRing(
    uint32_t              SimulationTime,
    uint32_t              threshold,
    uint32_t              membrane_potential,
    uint32_t              AmountOfCores,
    uint32_t              NeuronStart,
    uint32_t              NeuronTotal,
    hls::stream<ap_axiu<64, 0, 0, 0>> &syn_route_in,
    hls::stream<ap_axiu<64, 0, 0, 0>> &syn_forward_rt,
    hls::stream<stream512u_t> &synapse_stream,
    hls::stream<stream2048u_t> &spike_out);

extern "C" void kernel_mockup(
    uint32_t *hbm_in,
    uint32_t *hbm_out,
    float *hbm_out_float,
    uint32_t num_words
);

extern "C" void NeuroRing_singlestep(
    uint32_t                    *SynapseList,
    uint32_t                    *SpikeRecorder,
    uint32_t                     SimulationTime,
    float                        threshold,
    uint32_t                     AmountOfCores,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     DCstimStart,
    uint32_t                     DCstimTotal,
    float                        DCstimAmp);

#endif // NEURORING_H 