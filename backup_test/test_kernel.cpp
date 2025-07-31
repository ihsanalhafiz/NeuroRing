#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

// Simple test kernel: reads from HBM, streams, and writes back to HBM
// Demonstrates all typedefs, unions, and structs from NeuroRing.h

extern "C" void test_kernel(
    uint32_t *hbm_in,                // HBM input (m_axi)
    uint32_t *hbm_out,               // HBM output (m_axi)
    uint32_t  num_words,             // Number of 32-bit words to process
    hls::stream<stream512u_t> &to_next_kernel,   // AXI4-Stream out
    hls::stream<stream512u_t> &from_prev_kernel  // AXI4-Stream in
) {
#pragma HLS INTERFACE m_axi      port=hbm_in  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi      port=hbm_out offset=slave bundle=gmem1
#pragma HLS INTERFACE axis       port=to_next_kernel  bundle=AXIS_OUT
#pragma HLS INTERFACE axis       port=from_prev_kernel bundle=AXIS_IN
#pragma HLS INTERFACE s_axilite  port=hbm_in  bundle=control
#pragma HLS INTERFACE s_axilite  port=hbm_out bundle=control
#pragma HLS INTERFACE s_axilite  port=num_words bundle=control
#pragma HLS INTERFACE s_axilite  port=return   bundle=control

    // Example: Use all typedefs/unions/structs
    synapse_word_t syn_word = 0x123456789ABCDEF0ULL;
    spike_status_t spike_status = 0xDEADBEEF;
    DstID_t dst_id = 0xABCDEF;
    Delay_t delay = 0x3F;
    Weight_t weight = 1.23f;

    // Use stream types
    stream2048u_t s2048u;
    stream64u_t s64u;
    stream512u_t s512u;
    s2048u.data = 0x1;
    s64u.data = 0x2;
    s512u.data = 0x3;

    // Use pkt_union
    pkt_union pkt;
    pkt.parts.dst0 = dst_id;
    pkt.parts.delay0 = delay;
    pkt.parts.weight0 = weight;
    pkt.parts.dst1 = dst_id;
    pkt.parts.delay1 = delay;
    pkt.parts.weight1 = weight;
    pkt.parts.dst2 = dst_id;
    pkt.parts.delay2 = delay;
    pkt.parts.weight2 = weight;
    pkt.parts.dst3 = dst_id;
    pkt.parts.delay3 = delay;
    pkt.parts.weight3 = weight;
    pkt.parts.dst4 = dst_id;
    pkt.parts.delay4 = delay;
    pkt.parts.weight4 = weight;
    pkt.parts.dst5 = dst_id;
    pkt.parts.delay5 = delay;
    pkt.parts.weight5 = weight;
    pkt.parts.dst6 = dst_id;
    pkt.parts.delay6 = delay;
    pkt.parts.weight6 = weight;
    pkt.parts.dst7 = dst_id;
    pkt.parts.delay7 = delay;
    pkt.parts.weight7 = weight;

    // Use synapse_list_t
    synapse_list_t syn_list;
    syn_list.parts.DstID = dst_id;
    syn_list.parts.Delay = delay;
    syn_list.parts.Weight = weight;

    // Use structs
    StaticParams params;
    params.dt = 0.1f;
    params.tau = 10.0f;
    SpikeOut_t spike_out_vec;
    for (int i = 0; i < NEURON_NUM/32; i++) spike_out_vec.data[i] = 0;
    SynapseStream_t syn_stream_vec;
    for (int i = 0; i < 8; i++) syn_stream_vec.data[i] = syn_word;

    // Main logic: read from HBM, stream, and write back
    // (Assume num_words is a multiple of 16 for simplicity)
    for (uint32_t i = 0; i < num_words; i += 16) {
#pragma HLS PIPELINE II=1
        // Read 16 words from HBM
        hls::vector<uint32_t, 16> temp_read;
        for (int k = 0; k < 16; k++) {
            temp_read[k] = hbm_in[i + k];
        }
        // Pack into pkt_union and stream512u_t
        pkt_union pkt_out;
        for (int j = 0; j < 8; j++) {
            synapse_list_t syn;
            syn.parts.DstID = (temp_read[j*2] >> 8) & 0xFFFFFF;
            syn.parts.Delay = temp_read[j*2] & 0xFF;
            
            syn.parts.Weight = *((float*)&temp_read[j*2+1]);
            if(j==0) pkt_out.word.data0 = syn.word;
            if(j==1) pkt_out.word.data1 = syn.word;
            if(j==2) pkt_out.word.data2 = syn.word;
            if(j==3) pkt_out.word.data3 = syn.word;
            if(j==4) pkt_out.word.data4 = syn.word;
            if(j==5) pkt_out.word.data5 = syn.word;
            if(j==6) pkt_out.word.data6 = syn.word;
            if(j==7) pkt_out.word.data7 = syn.word;
        }
        to_next_kernel.write(pkt_out.packet);
    }

    // Read from previous kernel (or self-loop) and write back to HBM
    for (uint32_t i = 0; i < num_words/16; i++) {
#pragma HLS PIPELINE II=1
        stream512u_t pkt_in = from_prev_kernel.read();
        pkt_union pkt_unpack;
        pkt_unpack.packet = pkt_in;
        for (int j = 0; j < 8; j++) {
            synapse_list_t syn;
            if(j==0) syn.word = pkt_unpack.word.data0;
            if(j==1) syn.word = pkt_unpack.word.data1;
            if(j==2) syn.word = pkt_unpack.word.data2;
            if(j==3) syn.word = pkt_unpack.word.data3;
            if(j==4) syn.word = pkt_unpack.word.data4;
            if(j==5) syn.word = pkt_unpack.word.data5;
            if(j==6) syn.word = pkt_unpack.word.data6;
            if(j==7) syn.word = pkt_unpack.word.data7;
            // Unpack and write back as 2x32b words (DstID+Delay, Weight)
            uint32_t dst_delay = ((uint32_t)(syn.parts.DstID) << 8) | (uint32_t)(syn.parts.Delay);
            float w = syn.parts.Weight;
            hbm_out[i*16 + j*2] = dst_delay;
            hbm_out[i*16 + j*2 + 1] = *((uint32_t*)&w);
        }
    }
}
