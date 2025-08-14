#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

//====================================================================
//  2. AxonLoader – Fetch synapse lists when spikes occur
//====================================================================
extern "C" void AxonLoader(
    uint32_t                    *SpikeRecorder_SynapseList,
    int                     NeuronStart,
    int                     NeuronTotal,
    int                     DCstimStart,
    int                     DCstimTotal,
    int                     SimulationTime,
    int                     record_status,
    hls::stream<stream2048u_t>   &SpikeOutIn,
    hls::stream<stream2048u_t>    &SynapseStream)
{
    #pragma HLS INTERFACE m_axi port=SpikeRecorder_SynapseList offset=slave bundle=gmem0
    #pragma HLS INTERFACE axis port=SpikeOutIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseStream bundle=AXIS_OUT

    uint32_t DCstim_float[NEURON_NUM];
    uint32_t SynapseSize[NEURON_NUM];
    uint32_t UmemPot[NEURON_NUM];

    // Helper function to write packet to stream
    auto write_packet_to_stream = [](hls::stream<stream2048u_t>& stream, const stream2048u_t& packet) {
        bool write_status = false;
        while (!write_status) {
            write_status = stream.write_nb(packet);
        }
    };

    // read parameters from file
    for (int i = 0; i < NEURON_NUM; i++) {
        if (i < NeuronTotal) {
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
    }
    // send data of UmemPot to SynapseStream
    for (int i = 0; i < NEURON_NUM; i+=32) {
        if (i < NeuronTotal) {
            // UmemPot as weight, 0xFC as delay, and index as destination
            stream2048u_t packet;
            packet.data = 0;
            for (int j = 0; j < 32; j++) {
                int base_bit = 2047 - j * 64;
                packet.data.range(base_bit, base_bit - 23) = i+j+NeuronStart;
                packet.data.range(base_bit - 24, base_bit - 31) = 0xFC;
                packet.data.range(base_bit - 32, base_bit - 63) = UmemPot[i+j];
            }
            write_packet_to_stream(SynapseStream, packet);
        }
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
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 16; j++) {
                    SpikeRecorder_SynapseList[(t)*64 + i*16 + j] = spike_read.data.range(((i*16 + j)+1)*32-1, (i*16 + j)*32);
                }
            }
        }

        // Process each neuron that fired
        for (int i = 0; i < NEURON_NUM; i++) {
            if (i < NeuronTotal && spike_read.data.range(i, i) == 1) {  
                for (int j = 1; j < 9600/64; j++) {
                    if (j < SynapseSize[i]*2) {
                    // Read 16 synapses at once
                        stream2048u_t packet;
                        packet.data = 0;
                        for (int k = 0; k < 4; k++) {
                            hls::vector<uint32_t, 16> synapse_data;
                            #pragma HLS ARRAY_PARTITION variable=synapse_data complete dim=1
                            for (int p = 0; p < 16; p++) {
                                #pragma HLS UNROLL
                                synapse_data[p] = SpikeRecorder_SynapseList[i*SYNAPSE_LIST_SIZE + j*64 + k*16 + p + SYNAPSE_ARRAY_OFFSET];
                            }
                            packet.data.range(512*k + 31, 512*k) = synapse_data[15];
                            packet.data.range(512*k + 63, 512*k + 32) = synapse_data[14];
                            packet.data.range(512*k + 95, 512*k + 64) = synapse_data[13];
                            packet.data.range(512*k + 127, 512*k + 96) = synapse_data[12];
                            packet.data.range(512*k + 159, 512*k + 128) = synapse_data[11];
                            packet.data.range(512*k + 191, 512*k + 160) = synapse_data[10];
                            packet.data.range(512*k + 223, 512*k + 192) = synapse_data[9];
                            packet.data.range(512*k + 255, 512*k + 224) = synapse_data[8];
                            packet.data.range(512*k + 287, 512*k + 256) = synapse_data[7];
                            packet.data.range(512*k + 319, 512*k + 288) = synapse_data[6];
                            packet.data.range(512*k + 351, 512*k + 320) = synapse_data[5];
                            packet.data.range(512*k + 383, 512*k + 352) = synapse_data[4];
                            packet.data.range(512*k + 415, 512*k + 384) = synapse_data[3];
                            packet.data.range(512*k + 447, 512*k + 416) = synapse_data[2];
                            packet.data.range(512*k + 479, 512*k + 448) = synapse_data[1];
                            packet.data.range(512*k + 511, 512*k + 480) = synapse_data[0];
                        }
                        write_packet_to_stream(SynapseStream, packet);
                    }
                }
            }
        }

        // Handle DC stimulus window (default to kernel DCstimAmp for all neurons)
        if (t >= DCstimStart && t < DCstimStart + DCstimTotal) {
            for (int i = 0; i < NEURON_NUM; i += 32) {
                if (i < NeuronTotal) {
                    stream2048u_t packet;
                    packet.data = 0;
                    for (int j = 0; j < 32; j++) {
                        int base_bit = 2047 - j * 64;
                        packet.data.range(base_bit, base_bit - 23) = i+j+NeuronStart;
                        packet.data.range(base_bit - 24, base_bit - 31) = 0x00;
                        packet.data.range(base_bit - 32, base_bit - 63) = DCstim_float[i+j];
                    }
                    write_packet_to_stream(SynapseStream, packet);
                }
            }
        }

        // Send sync word
        stream2048u_t sync_packet;
        sync_packet.data = 0;
        uint32_t dst_delay_sync = (((NeuronStart) << 8) & 0xFFFFFF00) | 0xFE;
        sync_packet.data.range(2047, 2016) = dst_delay_sync;
        write_packet_to_stream(SynapseStream, sync_packet);
    }
}
