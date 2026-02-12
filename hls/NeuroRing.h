#ifndef NEURORING_H
#define NEURORING_H

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <ap_axi_sdata.h>
#include "hls_half.h"

#define NEURON_NUM 2816
#define SYNAPSE_TOTAL 7000
#define DELAY 64
#define SYNAPSE_LIST_SIZE SYNAPSE_TOTAL*2
#define SYNAPSE_ARRAY_OFFSET (100000 * (NEURON_NUM/32))

// AXI port data width
#define PTR_WIDTH 256
#define PTR_BYTE_WIDTH 32

typedef ap_uint<256>       synapse_list_t;
typedef ap_uint<32>        stream_weight_t;
typedef ap_uint<24>        DstID_t;
typedef ap_uint<8>         Delay_t;
typedef float               Weight_t;

// data, user, id, dest
typedef ap_axiu<256, 0, 0, 0> stream256u_t;

union float_to_uint32 {
    float f;
    uint32_t u;
};

extern "C" void NeuroRing(
    ap_uint<256>                 *SpikeRecorder_SynapseList,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     SimulationTime,
    uint32_t                     record_status,
    uint32_t                     CoreID,
    uint32_t                     AmountOfCores,
    hls::stream<stream256u_t>     &SpikeInWeight,
    hls::stream<stream256u_t>    &SynapseStream);

extern "C" void SynapseRouter(
    uint32_t              SimulationTime,
    uint32_t              AmountOfCores,
    uint32_t              NeuronStart,
    uint32_t              CoreID,
    hls::stream<stream256u_t> &syn_route_in,
    hls::stream<stream256u_t> &syn_forward_rt,
    hls::stream<stream256u_t> &synapse_stream,
    hls::stream<stream256u_t> &SpikeOutWeight);

extern "C" void SerialOut(
    hls::stream<stream256u_t>    &SynapseIn,
    hls::stream<ap_axiu<256, 0, 0, 0>>& data_output);

extern "C" void SerialIn(
    hls::stream<ap_axiu<256, 0, 0, 0>>& data_input,
    hls::stream<stream256u_t>    &SynapseOut);

void strm_issue (hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0>>& data_output,
                ap_uint<PTR_WIDTH> *data_input,
                unsigned int byte_size);

void strm_dump (hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0>>& data_input,
                ap_uint<PTR_WIDTH> *data_output,
                unsigned int byte_size);


#endif // NEURORING_H 