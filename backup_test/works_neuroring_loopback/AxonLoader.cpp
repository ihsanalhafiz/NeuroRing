#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1

#define NEURON_NUM 2048
#define SYNAPSE_LIST_SIZE 5000

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
    float                        DCstimAmp,
    uint32_t                     SimulationTime,
    hls::stream<stream2048u_t>   &SpikeOutIn,
    hls::stream<stream512u_t>    &SynapseStream)
{
    #pragma HLS INTERFACE m_axi      port=SynapseList   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=SpikeRecorder  offset=slave bundle=gmem1
    #pragma HLS INTERFACE axis port=SpikeOutIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseStream bundle=AXIS_OUT

    // TODO: implement burst HBM reads and stimulus injection.
    //  1) Wait for SpikeOut bitmap words from SomaEngine
    //  2) For each set bit – issue read to SynapseList
    //  3) Transmit resulting 8‑word bursts via SynapseStream
    //  4) Handle DC‑stimulus window
    //  5) Record spike monitor data into SpikeRecorder

    // Example skeleton read loop (blocking):
    read_status_loop: for (int t = 0; t < SimulationTime; t++) {
        ap_uint<2048> Spike_read = 0;
        for(int i = 0; i < (NeuronTotal/32); i++) {
            Spike_read.range((i+1)*32-1, i*32) = SpikeRecorder[t*64 + i];
        }
        for (int i = 0; i < NeuronTotal; i++) {
            if((Spike_read >> i) & 1) {
                uint32_t amount_of_synapses = SynapseList[i*SYNAPSE_LIST_SIZE];
                // make sure the size is divisible by 8
                if (amount_of_synapses % 16 != 0) amount_of_synapses += 16 - (amount_of_synapses % 16);
                for (int j = 0; j < amount_of_synapses/16; j++) {
                    hls::vector<uint32_t,16> temp_read;
                    for (int k = 0; k < 16; k++) {
                        temp_read[k] = SynapseList[i*SYNAPSE_LIST_SIZE + j*16 + k + 1];
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
                        write_status = SynapseStream.write_nb(packet);
                    }
                }
            }
        }
        // TODO: DC‑stimulus & sync word (all‑1 dstID)
        if(t >= DCstimStart && t < DCstimStart + DCstimTotal) {
            for(int i = NeuronStart; i < NeuronTotal; i+=8) {
                float weight0 = (float)(((i+0) < NeuronTotal) * DCstimAmp);
                float weight1 = (float)(((i+1) < NeuronTotal) * DCstimAmp);
                float weight2 = (float)(((i+2) < NeuronTotal) * DCstimAmp);
                float weight3 = (float)(((i+3) < NeuronTotal) * DCstimAmp);
                float weight4 = (float)(((i+4) < NeuronTotal) * DCstimAmp);
                float weight5 = (float)(((i+5) < NeuronTotal) * DCstimAmp);
                float weight6 = (float)(((i+6) < NeuronTotal) * DCstimAmp);
                float weight7 = (float)(((i+7) < NeuronTotal) * DCstimAmp);
                float_to_uint32 conv0;
                conv0.f = weight0;
                float_to_uint32 conv1;
                conv1.f = weight1;
                float_to_uint32 conv2;
                conv2.f = weight2;
                float_to_uint32 conv3;
                conv3.f = weight3;
                float_to_uint32 conv4;
                conv4.f = weight4;
                float_to_uint32 conv5;
                conv5.f = weight5;
                float_to_uint32 conv6;
                conv6.f = weight6;
                float_to_uint32 conv7;
                conv7.f = weight7;
                uint32_t dst_delay0 = ((i+0) << 8) & 0xFFFFFF00;
                uint32_t dst_delay1 = ((i+1) << 8) & 0xFFFFFF00;
                uint32_t dst_delay2 = ((i+2) << 8) & 0xFFFFFF00;
                uint32_t dst_delay3 = ((i+3) << 8) & 0xFFFFFF00;
                uint32_t dst_delay4 = ((i+4) << 8) & 0xFFFFFF00;
                uint32_t dst_delay5 = ((i+5) << 8) & 0xFFFFFF00;
                uint32_t dst_delay6 = ((i+6) << 8) & 0xFFFFFF00;
                uint32_t dst_delay7 = ((i+7) << 8) & 0xFFFFFF00;
    
                stream512u_t packet = {};
                packet.data.range(511, 480) = dst_delay0;
                packet.data.range(479, 448) = conv0.u;
                packet.data.range(447, 416) = dst_delay1;
                packet.data.range(415, 384) = conv1.u;
                packet.data.range(383, 352) = dst_delay2;
                packet.data.range(351, 320) = conv2.u;
                packet.data.range(319, 288) = dst_delay3;
                packet.data.range(287, 256) = conv3.u;
                packet.data.range(255, 224) = dst_delay4;
                packet.data.range(223, 192) = conv4.u;
                packet.data.range(191, 160) = dst_delay5;
                packet.data.range(159, 128) = conv5.u;
                packet.data.range(127, 96) = dst_delay6;
                packet.data.range(95, 64) = conv6.u;
                packet.data.range(63, 32) = dst_delay7;
                packet.data.range(31, 0) = conv7.u;
                bool write_status = false;
                while(!write_status) {
                    write_status = SynapseStream.write_nb(packet);
                }
            }
        }
        // sync word
        stream512u_t pkt_sync;
        pkt_sync.data = 0;
        uint32_t dst_delay_sync = (((NeuronStart)<< 8) & 0xFFFFFF00) | 0xFE;
        pkt_sync.data.range(511, 480) = dst_delay_sync;
        bool write_status = false;
        while(!write_status) {
            write_status = SynapseStream.write_nb(pkt_sync);
        }

        // read spike status
        stream2048u_t Spike;
        bool read_status = false;
        while(!read_status) {
            read_status = SpikeOutIn.read_nb(Spike);
        }

        for(int i = 0; i < 64; i++) {
            //SpikeRecorder[t*64 + i] = temp_SpikeRecorder[i];
            SpikeRecorder[(t+1)*64 + i] = Spike.data.range((i+1)*32-1,i*32);
        }
    }
}
