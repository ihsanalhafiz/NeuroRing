#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1

//====================================================================
//  2. AxonLoader – Fetch synapse lists when spikes occur
//====================================================================
extern "C" void AxonLoader(
    uint32_t                    *SpikeRecorder_SynapseList,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     DCstimStart,
    uint32_t                     DCstimTotal,
    uint32_t                     DCstimAmp,
    uint32_t                     SimulationTime,
    uint32_t                     record_status,
    hls::stream<stream2048u_t>   &SpikeOutIn,
    hls::stream<stream1024u_t>    &SynapseStream)
{
    #pragma HLS INTERFACE m_axi      port=SpikeRecorder_SynapseList   offset=slave bundle=gmem0
    #pragma HLS INTERFACE axis port=SpikeOutIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseStream bundle=AXIS_OUT

    // Helper function to create synapse packet from vector
    auto create_synapse_packet = [](const hls::vector<uint32_t, 32>& data) -> stream1024u_t {
        stream1024u_t packet = {};
        packet.data = 0;
        for (int k = 0; k < 32; k++) {
            packet.data.range(1023 - k*32, 992 - k*32) = data[k];
        }
        return packet;
    };

    uint32_t DCstim_float[NEURON_NUM];
    uint32_t SynapseSize[NEURON_NUM];
    uint32_t UmemPot[NEURON_NUM];

    // Helper function to write packet to stream
    auto write_packet_to_stream = [](hls::stream<stream1024u_t>& stream, const stream1024u_t& packet) {
        bool write_status = false;
        while (!write_status) {
            write_status = stream.write_nb(packet);
        }
    };

    // read parameters from file
    for (int i = 0; i < NeuronTotal; i++) {
        // read 16 data from SpikeRecorder_SynapseList
        hls::vector<uint32_t, 16> parameter_data;
        #pragma HLS ARRAY_PARTITION variable=parameter_data complete dim=1
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL
            parameter_data[k] = SpikeRecorder_SynapseList[i*SYNAPSE_LIST_SIZE + k + SYNAPSE_ARRAY_OFFSET];
        }
        SynapseSize[i] = parameter_data[0];
        DCstim_float[i] = parameter_data[1];
        UmemPot[i] = parameter_data[2];
    }

    // send data of UmemPot to SynapseStream
    for (int i = 0; i < NeuronTotal; i+=16) {
        // UmemPot as weight, 0xFC as delay, and index as destination
        stream1024u_t packet;
        packet.data = 0;
        for (int j = 0; j < 16; j++) {
            int base_bit = 1023 - j * 64;
            bool valid_neuron = (i+j < NeuronTotal);
            packet.data.range(base_bit, base_bit - 23) = valid_neuron ? i+j+NeuronStart : 0;
            packet.data.range(base_bit - 24, base_bit - 31) = 0xFC;
            packet.data.range(base_bit - 32, base_bit - 63) = valid_neuron ? UmemPot[i+j] : 0;
        }
        write_packet_to_stream(SynapseStream, packet);
    }

    // Main simulation loop
    read_status_loop: for (int t = 0; t < SimulationTime; t++) {
        // Read spike status from input stream
        stream2048u_t spike_read;
        bool read_status = false;
        while (!read_status) {
            read_status = SpikeOutIn.read_nb(spike_read);
        }
        if (record_status == 1) {
        // Record spike data
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    SpikeRecorder_SynapseList[(t)*64 + i*8 + j] = spike_read.data.range(((i*8 + j)+1)*32-1, (i*8 + j)*32);
                }
            }
        }

        // Process each neuron that fired
        for (int i = 0; i < NeuronTotal; i++) {
            if (spike_read.data.range(i, i) == 1) {                
                // Process synapses in chunks of 16
                for (int j = 1; j < (SynapseSize[i] + 31) / 32; j++) {
                    // Read 16 synapses at once
                    hls::vector<uint32_t, 32> synapse_data;
                    #pragma HLS ARRAY_PARTITION variable=synapse_data complete dim=1
                    for (int k = 0; k < 32; k++) {
                        #pragma HLS UNROLL
                        synapse_data[k] = SpikeRecorder_SynapseList[i*SYNAPSE_LIST_SIZE + j*32 + k + SYNAPSE_ARRAY_OFFSET];
                    }

                    // Create and send packet
                    stream1024u_t packet = create_synapse_packet(synapse_data);
                    write_packet_to_stream(SynapseStream, packet);
                }
            }
        }

        // Handle DC stimulus window (default to kernel DCstimAmp for all neurons)
        if (t >= DCstimStart && t < DCstimStart + DCstimTotal) {
            for (int i = NeuronStart; i < (int)(NeuronStart + NeuronTotal); i += 16) {
                // Use DCstimAmp uniformly; can be replaced with DCstim_float[i-NeuronStart] if per-neuron values are desired
                stream1024u_t packet;
                packet.data = 0;
                for (int j = 0; j < 16; j++) {
                    int base_bit = 1023 - j * 64;
                    bool valid_neuron = (i+j < (NeuronStart + NeuronTotal));
                    packet.data.range(base_bit, base_bit - 23) = valid_neuron ? i+j : 0;
                    packet.data.range(base_bit - 24, base_bit - 31) = 0x00;
                    packet.data.range(base_bit - 32, base_bit - 63) = valid_neuron ? DCstim_float[(i+j)-NeuronStart] : 0;
                }
                write_packet_to_stream(SynapseStream, packet);
            }
        }

        // Send sync word
        stream1024u_t sync_packet;
        sync_packet.data = 0;
        uint32_t dst_delay_sync = (((NeuronStart) << 8) & 0xFFFFFF00) | 0xFE;
        sync_packet.data.range(1023, 992) = dst_delay_sync;
        write_packet_to_stream(SynapseStream, sync_packet);

    }
}
