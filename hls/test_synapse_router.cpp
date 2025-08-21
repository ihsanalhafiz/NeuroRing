#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include "NeuroRing.h"

// Test function to verify SynapseRouter functionality
void test_synapse_router() {
    hls::stream<stream512u_t> input_stream;
    hls::stream<stream512u_t> output_stream;
    
    // Test case 1: Create a packet with destinations in different ranges
    stream512u_t test_packet1;
    test_packet1.data = 0;
    
    // Entry 0: destination 1000 (slot 0)
    test_packet1.data.range(511, 488) = 1000;      // dst
    test_packet1.data.range(487, 480) = 0x10;      // delay
    test_packet1.data.range(479, 448) = 0x12345678; // weight
    
    // Entry 1: destination 3000 (slot 1)
    test_packet1.data.range(447, 424) = 3000;      // dst
    test_packet1.data.range(423, 416) = 0x20;      // delay
    test_packet1.data.range(415, 384) = 0x87654321; // weight
    
    // Entry 2: destination 5000 (slot 2)
    test_packet1.data.range(383, 360) = 5000;      // dst
    test_packet1.data.range(359, 352) = 0x30;      // delay
    test_packet1.data.range(351, 320) = 0xABCDEF01; // weight
    
    // Entry 3: destination 7000 (slot 3)
    test_packet1.data.range(319, 296) = 7000;      // dst
    test_packet1.data.range(295, 288) = 0x40;      // delay
    test_packet1.data.range(287, 256) = 0xFEDCBA98; // weight
    
    // Entry 4: destination 1500 (slot 0)
    test_packet1.data.range(255, 232) = 1500;      // dst
    test_packet1.data.range(231, 224) = 0x50;      // delay
    test_packet1.data.range(223, 192) = 0x11111111; // weight
    
    // Entry 5: destination 3500 (slot 1)
    test_packet1.data.range(191, 168) = 3500;      // dst
    test_packet1.data.range(167, 160) = 0x60;      // delay
    test_packet1.data.range(159, 128) = 0x22222222; // weight
    
    // Entry 6: destination 5500 (slot 2)
    test_packet1.data.range(127, 104) = 5500;      // dst
    test_packet1.data.range(103, 96) = 0x70;       // delay
    test_packet1.data.range(95, 64) = 0x33333333;  // weight
    
    // Entry 7: destination 7500 (slot 3)
    test_packet1.data.range(63, 40) = 7500;        // dst
    test_packet1.data.range(39, 32) = 0x80;        // delay
    test_packet1.data.range(31, 0) = 0x44444444;   // weight
    
    test_packet1.last = 0;
    
    // Write test packet to input stream
    input_stream.write(test_packet1);
    
    // Test case 2: Create another packet to fill slots
    stream512u_t test_packet2;
    test_packet2.data = 0;
    
    // Fill remaining slots with more entries
    for (int i = 0; i < 8; i++) {
        int base_bit = 511 - i * 64;
        int dst = 100 + i * 100;  // destinations 100, 200, 300, etc. (all slot 0)
        test_packet2.data.range(base_bit, base_bit - 23) = dst;
        test_packet2.data.range(base_bit - 24, base_bit - 31) = 0x90 + i;
        test_packet2.data.range(base_bit - 32, base_bit - 63) = 0x50000000 + i;
    }
    test_packet2.last = 0;
    
    // Write second test packet
    input_stream.write(test_packet2);
    
    // Test case 3: Create sync packet (last flag)
    stream512u_t sync_packet;
    sync_packet.data = 0;
    // Set delay to 0xFE to indicate last packet
    sync_packet.data.range(487, 480) = 0xFE;
    sync_packet.last = 1;
    
    // Write sync packet
    input_stream.write(sync_packet);
    
    std::cout << "Test packets created and written to input stream." << std::endl;
    std::cout << "Packet 1: Mixed destinations across all 4 slots" << std::endl;
    std::cout << "Packet 2: All destinations in slot 0 (to fill slot 0)" << std::endl;
    std::cout << "Packet 3: Sync packet with last flag" << std::endl;
    
    // Note: In a real HLS simulation, you would call SynapseRouter here
    // SynapseRouter(input_stream, output_stream);
    
    std::cout << "Test setup complete. Run HLS simulation to test SynapseRouter kernel." << std::endl;
}

int main() {
    test_synapse_router();
    return 0;
}
