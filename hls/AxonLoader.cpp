#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1

#define NEURON_NUM 2048
#define SYNAPSE_LIST_SIZE 10000

//====================================================================
//  2. AxonLoader – Fetch synapse lists when spikes occur
//====================================================================
extern "C" void AxonLoader(
    uint32_t                    *SynapseList,
    uint32_t                    *SpikeRecorder,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     DCstimStart,
    uint32_t                     DCstimTotal,
    uint32_t                     DCstimAmp,
    uint32_t                     SimulationTime,
    uint32_t                     record_status,
    hls::stream<stream2048u_t>   &SpikeOutIn,
    hls::stream<stream512u_t>    &SynapseStream)
{
    #pragma HLS INTERFACE m_axi      port=SynapseList   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=SpikeRecorder  offset=slave bundle=gmem1
    #pragma HLS INTERFACE axis port=SpikeOutIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseStream bundle=AXIS_OUT

    // Helper function to create synapse packet from vector
    auto create_synapse_packet = [](const hls::vector<uint32_t, 16>& data) -> stream512u_t {
        stream512u_t packet = {};
        packet.data = 0;
        for (int k = 0; k < 16; k++) {
            packet.data.range(511 - k*32, 480 - k*32) = data[k];
        }
        return packet;
    };

    // Helper function to create DC stimulus packet
    auto create_dc_stimulus_packet = [](uint32_t base_neuron, uint32_t neuron_total, 
                                       uint32_t dc_stim_amp) -> stream512u_t {
        stream512u_t packet = {};
        packet.data = 0;
        
        for (int offset = 0; offset < 8; offset++) {
            uint32_t neuron_idx = base_neuron + offset;
            bool valid_neuron = (neuron_idx < neuron_total);
            
            // Convert float weight to uint32
            uint32_t weight = valid_neuron ? dc_stim_amp : 0;
            
            // Create destination and delay
            uint32_t dst_delay = (neuron_idx << 8) & 0xFFFFFF00;
            
            // Pack into packet (8 neurons per packet)
            packet.data.range(511 - offset*64, 480 - offset*64) = dst_delay;
            packet.data.range(479 - offset*64, 448 - offset*64) = weight;
        }
        
        return packet;
    };

    // Helper function to write packet to stream
    auto write_packet_to_stream = [](hls::stream<stream512u_t>& stream, const stream512u_t& packet) {
        bool write_status = false;
        while (!write_status) {
            write_status = stream.write_nb(packet);
        }
    };

    // Main simulation loop
    read_status_loop: for (int t = 0; t < SimulationTime; t++) {
        // Read spike status from input stream
        stream2048u_t spike_data;
        bool read_status = false;
        while (!read_status) {
            read_status = SpikeOutIn.read_nb(spike_data);
        }
        if (record_status == 1) {
        // Record spike data
            for (int i = 0; i < 64; i++) {
                SpikeRecorder[(t)*64 + i] = spike_data.data.range((i+1)*32-1, i*32);
            }
        }

        // Process each neuron that fired
        for (int i = 0; i < NeuronTotal; i++) {
            if (spike_read.range(i, i) == 1) {
                // Read synapse count and ensure it's divisible by 16
                uint32_t synapse_count = SynapseList[i*SYNAPSE_LIST_SIZE];
                // Process synapses in chunks of 16
                for (int j = 0; j < (synapse_count + 15) / 16; j++) {
                    // Read 16 synapses at once
                    hls::vector<uint32_t, 16> synapse_data;
                    #pragma HLS ARRAY_PARTITION variable=synapse_data complete dim=1
                    for (int k = 0; k < 16; k++) {
                        synapse_data[k] = SynapseList[i*SYNAPSE_LIST_SIZE + j*16 + k + 1];
                    }
                    
                    // Create and send packet
                    stream512u_t packet = create_synapse_packet(synapse_data);
                    write_packet_to_stream(SynapseStream, packet);
                }
            }
        }

        // Handle DC stimulus window
        if (t >= DCstimStart && t < DCstimStart + DCstimTotal) {
            for (int i = NeuronStart; i < NeuronTotal; i += 8) {
                stream512u_t packet = create_dc_stimulus_packet(i, NeuronTotal, DCstimAmp);
                write_packet_to_stream(SynapseStream, packet);
            }
        }

        // Send sync word
        stream512u_t sync_packet;
        sync_packet.data = 0;
        uint32_t dst_delay_sync = (((NeuronStart) << 8) & 0xFFFFFF00) | 0xFE;
        sync_packet.data.range(511, 480) = dst_delay_sync;
        write_packet_to_stream(SynapseStream, sync_packet);

    }
}
