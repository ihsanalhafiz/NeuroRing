#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1

//====================================================================
//  SynapseRouter – Routes synapse data to 4 slots based on destination
//====================================================================
extern "C" void Router(
    hls::stream<stream512u_t>    &SynapseStream,
    hls::stream<stream512u_t>    &SynapseOut)
{
    #pragma HLS INTERFACE axis port=SynapseStream bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseOut bundle=AXIS_OUT
    #pragma HLS INTERFACE ap_ctrl_none port=return

    const int id = 0;

    // Define slot buffers - each slot can hold 8 synapse entries (512 bits)
    ap_uint<512> slot_buffer[4];
    #pragma HLS ARRAY_PARTITION variable=slot_buffer complete dim=1
    
    // Slot counters - track how many synapse entries are in each slot
    ap_uint<4> slot_count[4];
    #pragma HLS ARRAY_PARTITION variable=slot_count complete dim=1

    DstID_t dst[8];
    Delay_t delay[8];
    Weight_t weight[8];
    #pragma HLS ARRAY_PARTITION variable=dst complete dim=1
    #pragma HLS ARRAY_PARTITION variable=delay complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=1

    DstID_t dst1[8];
    Delay_t delay1[8];
    Weight_t weight1[8];
    #pragma HLS ARRAY_PARTITION variable=dst1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=delay1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight1 complete dim=1

    DstID_t dst2[8];
    Delay_t delay2[8];
    Weight_t weight2[8];
    #pragma HLS ARRAY_PARTITION variable=dst2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=delay2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight2 complete dim=1

    DstID_t dst3[8];
    Delay_t delay3[8];
    Weight_t weight3[8];
    #pragma HLS ARRAY_PARTITION variable=dst3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=delay3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight3 complete dim=1
    
    // Initialize slot buffers and counters to 0
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        slot_buffer[i] = 0;
        slot_count[i] = 0;
    }
    for(int i = 0; i < 8; i++) {
        dst[i] = 0;
        delay[i] = 0;
        weight[i] = 0;
        dst1[i] = 0;
        delay1[i] = 0;
        weight1[i] = 0;
        dst2[i] = 0;
        delay2[i] = 0;
        weight2[i] = 0;
        dst3[i] = 0;
        delay3[i] = 0;
        weight3[i] = 0;
    }

    // Free-running loop
    while (true) {
        #pragma HLS PIPELINE II=1 rewind
        // Check if any slot is full and output it
        for (int slot = 0; slot < 4; slot++) {
            #pragma HLS UNROLL
            if (slot_count[slot] >= 8) {
                // Create output packet
                stream512u_t output_packet;
                for(int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
                    output_packet.data.range(base_bit, base_bit - 23) = dst[i];
                    output_packet.data.range(base_bit - 24, base_bit - 31) = delay[i];
                    output_packet.data.range(base_bit - 32, base_bit - 63) = weight[i];
                    dst[i] = 0;
                    delay[i] = 0;
                    weight[i] = 0;
                }
                output_packet.last = 0;  // Not the last packet
                output_packet.id = id;
                output_packet.dest = slot;  // ID and DEST encoded in TID/TDEST
                
                // Write to output stream
                SynapseOut.write(output_packet);
                
                // Reset slot buffer and counter
                slot_buffer[slot] = 0;
                slot_count[slot] = 0;
            }
        }
        
        // Try to read from input stream
        stream512u_t input_packet;
        if (SynapseStream.read_nb(input_packet)) {
            if (input_packet.last) {
                // Output all non-empty slots with last flag
                for (int slot = 0; slot < 4; slot++) {
                    #pragma HLS UNROLL
                    // Create output packet
                    stream512u_t output_packet;
                    for(int i = 0; i < 8; i++) {
                        #pragma HLS UNROLL
                        int base_bit = 511 - i * 64;
                        output_packet.data.range(base_bit, base_bit - 23) = dst[i];
                        output_packet.data.range(base_bit - 24, base_bit - 31) = delay[i];
                        output_packet.data.range(base_bit - 32, base_bit - 63) = weight[i];
                        dst[i] = 0;
                        delay[i] = 0;
                        weight[i] = 0;
                    }
                    output_packet.last = 1;  // Last packet flag
                    output_packet.id = id;
                    output_packet.dest = slot;  // ID and DEST encoded in TID/TDEST
                        
                    // Write to output stream
                    SynapseOut.write(output_packet);
                    // Reset slot buffer and counter
                    slot_buffer[slot] = 0;
                    slot_count[slot] = 0;
                }
            } else {
                // Process each of the 8 synapse entries in the packet
                for (int i = 0; i < 8; i++) {
                    #pragma HLS PIPELINE II=1 
                    int base_bit = 511 - i * 64;
                    
                    // Extract synapse data
                    ap_uint<24> dst = input_packet.data.range(base_bit, base_bit - 23);
                    ap_uint<8> delay = input_packet.data.range(base_bit - 24, base_bit - 31);
                    ap_uint<32> weight = input_packet.data.range(base_bit - 32, base_bit - 63);
                    
                    // Determine which slot this synapse should go to based on destination
                    int target_slot;
                    if (dst >= 1 && dst <= 2048) {
                        target_slot = 0;
                        if(slot_count[target_slot] < 8) {
                            dst[slot_count[target_slot]] = dst;
                            delay[slot_count[target_slot]] = delay;
                            weight[slot_count[target_slot]] = weight;
                            slot_count[target_slot]++;
                        }
                    } else if (dst >= 2049 && dst <= 4096) {
                        target_slot = 1;
                        if(slot_count[target_slot] < 8) {
                            dst1[slot_count[target_slot]] = dst;
                            delay1[slot_count[target_slot]] = delay;
                            weight1[slot_count[target_slot]] = weight;
                            slot_count[target_slot]++;
                        }
                    } else if (dst >= 4097 && dst <= 6144) {
                        target_slot = 2;
                        if(slot_count[target_slot] < 8) {
                            dst2[slot_count[target_slot]] = dst;
                            delay2[slot_count[target_slot]] = delay;
                            weight2[slot_count[target_slot]] = weight;
                            slot_count[target_slot]++;
                        }
                    } else if (dst >= 6145 && dst <= 8192) {
                        target_slot = 3;
                        if(slot_count[target_slot] < 8) {
                            dst3[slot_count[target_slot]] = dst;
                            delay3[slot_count[target_slot]] = delay;
                            weight3[slot_count[target_slot]] = weight;
                            slot_count[target_slot]++;
                        }
                    } else {
                        // Invalid destination, skip this entry
                        continue;
                    }
                }
            }
        }
    }
}
