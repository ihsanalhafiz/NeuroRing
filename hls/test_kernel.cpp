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
        stream512u_t packet = {};
        packet.data.range(511, 480) = temp_read[0];
        packet.data.range(479, 448) = temp_read[1];
        packet.data.range(447, 416) = temp_read[2];
        packet.data.range(415, 384) = temp_read[3];
        packet.data.range(383, 352) = temp_read[4];
        packet.data.range(351, 320) = temp_read[5];
        packet.data.range(319, 288) = temp_read[6];
        packet.data.range(287, 256) = temp_read[7];
        packet.data.range(255, 224) = temp_read[8];
        packet.data.range(223, 192) = temp_read[9];
        packet.data.range(191, 160) = temp_read[10];
        packet.data.range(159, 128) = temp_read[11];
        packet.data.range(127, 96) = temp_read[12];
        packet.data.range(95, 64) = temp_read[13];
        packet.data.range(63, 32) = temp_read[14];
        packet.data.range(31, 0) = temp_read[15];

        bool write_status = false;
        while(!write_status) {
            write_status = to_next_kernel.write_nb(packet);
        }
    }

    // Read from previous kernel (or self-loop) and write back to HBM
    for (uint32_t i = 0; i < num_words/16; i++) {
#pragma HLS PIPELINE II=1
        bool read_status = false;
        stream512u_t pkt_in;
        while(!read_status) {
            read_status = from_prev_kernel.read_nb(pkt_in);
        }
        for (int k = 0; k < 16; k++) {
            hbm_out[i*16 + k] = pkt_in.data.range(511 - 32*k, 480 - 32*k);
        }
    }
}
