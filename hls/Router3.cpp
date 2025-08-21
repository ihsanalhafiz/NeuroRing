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
extern "C" void Router3(
    hls::stream<stream512u_t>    &SynapseStream,
    hls::stream<stream512u_t>    &SynapseOut)
{
    #pragma HLS INTERFACE axis port=SynapseStream bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseOut bundle=AXIS_OUT
    #pragma HLS INTERFACE ap_ctrl_none port=return

    const int id = 3;

    // Define slot buffers - each slot can hold 8 synapse entries (512 bits)
    ap_uint<512> slot_buffer[4];
    #pragma HLS ARRAY_PARTITION variable=slot_buffer complete dim=1
    
    // Slot counters - track how many synapse entries are in each slot
    ap_uint<4> slot_count[4];
    #pragma HLS ARRAY_PARTITION variable=slot_count complete dim=1
    
    // Initialize slot buffers and counters to 0
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        slot_buffer[i] = 0;
        slot_count[i] = 0;
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
                output_packet.data = slot_buffer[slot];
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
                    output_packet.data = slot_buffer[slot];
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
                    } else if (dst >= 2049 && dst <= 4096) {
                        target_slot = 1;
                    } else if (dst >= 4097 && dst <= 6144) {
                        target_slot = 2;
                    } else if (dst >= 6145 && dst <= 8192) {
                        target_slot = 3;
                    } else {
                        // Invalid destination, skip this entry
                        continue;
                    }
                    
                    // Check if target slot has space
                    if (slot_count[target_slot] < 8) {
                        // Calculate position in slot buffer
                        int pos_in_slot = slot_count[target_slot];
                        int slot_base_bit = 511 - pos_in_slot * 64;
                        
                        // Pack synapse data into slot buffer
                        slot_buffer[target_slot].range(slot_base_bit, slot_base_bit - 23) = dst;
                        slot_buffer[target_slot].range(slot_base_bit - 24, slot_base_bit - 31) = delay;
                        slot_buffer[target_slot].range(slot_base_bit - 32, slot_base_bit - 63) = weight;
                        
                        // Increment slot counter
                        slot_count[target_slot]++;
                    }
                }
            }
        }
    }
}
