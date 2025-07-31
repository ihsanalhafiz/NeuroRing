#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>
#include "NeuroRing.h"

// Producer: Reads from HBM, packs into stream512u_t, writes to stream
void producer(
    uint32_t *hbm_in,
    uint32_t num_words,
    hls::stream<stream512u_t> &out_stream
) {
#pragma HLS INLINE off
    for (uint32_t i = 0; i < num_words; i += 16) {
#pragma HLS PIPELINE II=1
        hls::vector<uint32_t, 16> temp_read;
        for (int k = 0; k < 16; k++) {
            temp_read[k] = hbm_in[i + k];
        }
        stream512u_t packet = {};
        
        // Build 512-bit packet: words 0-15 in order
        packet.data = (ap_uint<512>)temp_read[0] << 480 | 
                     (ap_uint<512>)temp_read[1] << 448 | 
                     (ap_uint<512>)temp_read[2] << 416 | 
                     (ap_uint<512>)temp_read[3] << 384 |
                     (ap_uint<512>)temp_read[4] << 352 | 
                     (ap_uint<512>)temp_read[5] << 320 | 
                     (ap_uint<512>)temp_read[6] << 288 | 
                     (ap_uint<512>)temp_read[7] << 256 |
                     (ap_uint<512>)temp_read[8] << 224 | 
                     (ap_uint<512>)temp_read[9] << 192 | 
                     (ap_uint<512>)temp_read[10] << 160 | 
                     (ap_uint<512>)temp_read[11] << 128 |
                     (ap_uint<512>)temp_read[12] << 96 | 
                     (ap_uint<512>)temp_read[13] << 64 | 
                     (ap_uint<512>)temp_read[14] << 32 | 
                     (ap_uint<512>)temp_read[15];
        
        out_stream.write(packet);
    }
}

// Consumer: Reads from stream512u_t, unpacks, writes to HBM
void consumer(
    hls::stream<stream512u_t> &in_stream,
    uint32_t *hbm_out,
    float *hbm_out_float,
    uint32_t num_words
) {
#pragma HLS INLINE off
    for (uint32_t i = 0; i < num_words/16; i++) {
#pragma HLS PIPELINE II=1
        stream512u_t pkt_in = in_stream.read();
        // Extract 16 words from 512-bit packet in order
        for (int s = 0; s < 16; s++) {
            // Extract 32-bit word at position s
            uint32_t word = (pkt_in.data >> (32 * (15 - s))) & 0xFFFFFFFF;
            hbm_out[i*16 + s] = word;
            
            // For odd indices (weights), also store in float buffer
            if (s % 2 == 1) {
                float_to_uint32 weight_union;
                weight_union.u = word;
                hbm_out_float[i*8 + s/2] = weight_union.f;
            }
        }
    }
}

// Top-level kernel: connects producer and consumer with dataflow
extern "C" void kernel_mockup(
    uint32_t *hbm_in,
    uint32_t *hbm_out,
    float *hbm_out_float,
    uint32_t num_words
) {
#pragma HLS INTERFACE m_axi      port=hbm_in  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi      port=hbm_out offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite  port=num_words bundle=control
#pragma HLS INTERFACE s_axilite  port=return   bundle=control

#pragma HLS DATAFLOW
    static hls::stream<stream512u_t> stream_ch("stream_ch");
#pragma HLS STREAM variable=stream_ch depth=8
    producer(hbm_in, num_words, stream_ch);
    consumer(stream_ch, hbm_out, hbm_out_float, num_words);
} 