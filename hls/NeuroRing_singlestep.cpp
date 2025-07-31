//============================================================
//  NeuroRing ‒ Scalable HPC FPGA SNN Accelerator (Skeleton)
//  --------------------------------------------------------
//  Skeleton generated 14 Jul 2025 for Vitis HLS 2024.2+
//  --------------------------------------------------------
//  4 Sub‑kernels
//    1. AxonLoader
//    2. SynapseRouter
//    3. DendriteDelay
//    4. SomaEngine
//============================================================
//  NOTES
//  -----
//  •  This file is meant as a *starting point* only ‒ all algorithmic
//     details are condensed to TODO tags. Replace them with your own
//     implementation logic.
//  •  The top‑level kernel uses DATAFLOW to run every sub‑kernel in
//     parallel, communicating through AXI4‑Stream channels.
//  •  Update interface bundles, depths and types to match your board
//     constraints (HBM, DDR, FIFO sizes, etc.).
//============================================================

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1

#define NEURON_NUM_X 2048
#define SYNAPSE_LIST_SIZE_X 100

//--------------------------------------------------------------------
//  Top‑level kernel ‒ integrates all sub‑kernels using DATAFLOW
//--------------------------------------------------------------------
extern "C" void NeuroRing_singlestep(
    uint32_t                    *SynapseList,
    uint32_t                    *SpikeRecorder,
    uint32_t                     SimulationTime,
    float                        threshold,
    uint32_t                     AmountOfCores,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     DCstimStart,
    uint32_t                     DCstimTotal,
    float                        DCstimAmp)
{
    hls::vector<uint32_t, NEURON_NUM_X/32> spike_recorder = 0;
    hls::stream<stream2048u_t> SpikeOut;
    hls::stream<stream512u_t> SynapseStream;
    stream2048u_t Spike_send;
    Spike_send.data = 0; // Make sure to clear it first!
    for (int i = 0; i < (NeuronTotal/32); i++) {
        // Place each 32-bit word at the correct offset
        Spike_send.data.range((i+1)*32-1, i*32) = SpikeRecorder[i];
    }
    SpikeOut.write(Spike_send);
    printf("SpikeRecorder done\n");

// AxonLoader ------------------------------------------------------------------------------
    printf("AxonLoader start\n");
    stream2048u_t Spike;
    bool read_status_spike = false;
    while(!read_status_spike) {
        read_status_spike = SpikeOut.read_nb(Spike);
    }

    for (int i = 0; i < NeuronTotal; i++) {
        if((Spike.data >> i) & 1) {
            uint32_t amount_of_synapses = SynapseList[i*SYNAPSE_LIST_SIZE_X];
            // make sure the size is divisible by 8
            if (amount_of_synapses % 16 != 0) amount_of_synapses += 16 - (amount_of_synapses % 16);
            for (int j = 0; j < amount_of_synapses/16; j++) {
                hls::vector<uint32_t,16> temp_read;
                for (int k = 0; k < 16; k++) {
                    temp_read[k] = SynapseList[i*SYNAPSE_LIST_SIZE_X + j*16 + k + 1];
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
                bool write_status_packet = false;
                while(!write_status_packet) {
                    write_status_packet = SynapseStream.write_nb(packet);
                }

            }
        }
    }
    printf("AxonLoader done\n");

// DCstim ------------------------------------------------------------------------------
    printf("DCstim start\n");
    if(DCstimStart >= 0 && DCstimStart < DCstimTotal) {
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
            bool write_status_packet = false;
            while(!write_status_packet) {
                write_status_packet = SynapseStream.write_nb(packet);
            }
        }
    }
    printf("DCstim done\n");

    stream512u_t pkt_sync;
    pkt_sync.data = 0;
    uint32_t dst_delay_sync = (((NeuronStart)<< 8) & 0xFFFFFF00) | 0xFE;
    pkt_sync.data.range(511, 480) = dst_delay_sync;
    bool write_status_sync = false;
    while(!write_status_sync) {
        write_status_sync = SynapseStream.write_nb(pkt_sync);
    }

// SynapseRouter ------------------------------------------------------------------------------
    printf("SynapseRouter start\n");
    hls::stream<ap_axiu<64, 0, 0, 0>> SynForwardRoute;
    hls::stream<synapse_word_t> SynForward;
    bool axon_done = false;
    bool prev_done = false;
    uint32_t coreDone = 0;
    DstID_t dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
    Weight_t weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7;
    Delay_t delay0, delay1, delay2, delay3, delay4, delay5, delay6, delay7;
    while (!(axon_done && prev_done)) {
    #pragma HLS PIPELINE II=1 rewind
        stream512u_t pkt;
        bool have_pkt = SynapseStream.read_nb(pkt);
        if (have_pkt) {
            dst0 = pkt.data.range(511, 488);
            delay0 = pkt.data.range(487, 480);
            float_to_uint32 weight_conv0;
            weight_conv0.u = pkt.data.range(479, 448);

            dst1 = pkt.data.range(447, 424);
            delay1 = pkt.data.range(423, 416);
            float_to_uint32 weight_conv1;
            weight_conv1.u = pkt.data.range(415, 384);

            dst2 = pkt.data.range(383, 360);
            delay2 = pkt.data.range(359, 352);
            float_to_uint32 weight_conv2;
            weight_conv2.u = pkt.data.range(351, 320);

            dst3 = pkt.data.range(319, 296);
            delay3 = pkt.data.range(295, 288);
            float_to_uint32 weight_conv3;
            weight_conv3.u = pkt.data.range(287, 256);
            
            dst4 = pkt.data.range(255, 232);
            delay4 = pkt.data.range(231, 224);
            float_to_uint32 weight_conv4;
            weight_conv4.u = pkt.data.range(223, 192);

            dst5 = pkt.data.range(191, 168);
            delay5 = pkt.data.range(167, 160);
            float_to_uint32 weight_conv5;
            weight_conv5.u = pkt.data.range(159, 128);

            dst6 = pkt.data.range(127, 104);
            delay6 = pkt.data.range(103, 96);
            float_to_uint32 weight_conv6;
            weight_conv6.u = pkt.data.range(95, 64);

            dst7 = pkt.data.range(63, 40);
            delay7 = pkt.data.range(39, 32);
            float_to_uint32 weight_conv7;
            weight_conv7.u = pkt.data.range(31, 0);

            printf("weight0: %f, weight1: %f, weight2: %f, weight3: %f, weight4: %f, weight5: %f, weight6: %f, weight7: %f\n", weight_conv0.f, weight_conv1.f, weight_conv2.f, weight_conv3.f, weight_conv4.f, weight_conv5.f, weight_conv6.f, weight_conv7.f);

            if (delay0 == 0xFE) {
                axon_done = true;
                ap_axiu<64, 0, 0, 0> temp_sync;
                //temp_sync.data = ((ap_uint<64>)dst0 << 40) | ((ap_uint<64>)delay0 << 32) | ((ap_uint<64>)weight_conv0.u);
                temp_sync.data.range(63, 40) = dst0;
                temp_sync.data.range(39, 32) = delay0;
                temp_sync.data.range(31, 0) = weight_conv0.u;
                bool write_status_sync = false;
                while(!write_status_sync) {
                    write_status_sync = SynForwardRoute.write_nb(temp_sync);
                }
            } else {
                synapse_word_t temp0;
                //temp0.data = ((ap_uint<64>)dst0 << 40) | ((ap_uint<64>)delay0 << 32) | ((ap_uint<64>)weight_conv0.u);
                temp0.range(63, 40) = dst0;
                temp0.range(39, 32) = delay0;
                temp0.range(31, 0) = weight_conv0.u;
                synapse_word_t temp1;
                //temp1.data = ((ap_uint<64>)dst1 << 40) | ((ap_uint<64>)delay1 << 32) | ((ap_uint<64>)weight_conv1.u);
                temp1.range(63, 40) = dst1;
                temp1.range(39, 32) = delay1;
                temp1.range(31, 0) = weight_conv1.u;
                synapse_word_t temp2;
                //temp2.data = ((ap_uint<64>)dst2 << 40) | ((ap_uint<64>)delay2 << 32) | ((ap_uint<64>)weight_conv2.u);
                temp2.range(63, 40) = dst2;
                temp2.range(39, 32) = delay2;
                temp2.range(31, 0) = weight_conv2.u;
                synapse_word_t temp3;
                //temp3.data = ((ap_uint<64>)dst3 << 40) | ((ap_uint<64>)delay3 << 32) | ((ap_uint<64>)weight_conv3.u);
                temp3.range(63, 40) = dst3;
                temp3.range(39, 32) = delay3;
                temp3.range(31, 0) = weight_conv3.u;
                synapse_word_t temp4;
                //temp4.data = ((ap_uint<64>)dst4 << 40) | ((ap_uint<64>)delay4 << 32) | ((ap_uint<64>)weight_conv4.u);
                temp4.range(63, 40) = dst4;
                temp4.range(39, 32) = delay4;
                temp4.range(31, 0) = weight_conv4.u;
                synapse_word_t temp5;
                //temp5.data = ((ap_uint<64>)dst5 << 40) | ((ap_uint<64>)delay5 << 32) | ((ap_uint<64>)weight_conv5.u);
                temp5.range(63, 40) = dst5;
                temp5.range(39, 32) = delay5;
                temp5.range(31, 0) = weight_conv5.u;
                synapse_word_t temp6;
                //temp6.data = ((ap_uint<64>)dst6 << 40) | ((ap_uint<64>)delay6 << 32) | ((ap_uint<64>)weight_conv6.u);
                temp6.range(63, 40) = dst6;
                temp6.range(39, 32) = delay6;
                temp6.range(31, 0) = weight_conv6.u;
                synapse_word_t temp7;
                //temp7.data = ((ap_uint<64>)dst7 << 40) | ((ap_uint<64>)delay7 << 32) | ((ap_uint<64>)weight_conv7.u);
                temp7.range(63, 40) = dst7;
                temp7.range(39, 32) = delay7;
                temp7.range(31, 0) = weight_conv7.u;
                printf("dst0: %u, dst1: %u, dst2: %u, dst3: %u, dst4: %u, dst5: %u, dst6: %u, dst7: %u\n", dst0.to_uint(), dst1.to_uint(), dst2.to_uint(), dst3.to_uint(), dst4.to_uint(), dst5.to_uint(), dst6.to_uint(), dst7.to_uint());
                if (dst0 >= NeuronStart && dst0 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp0);
                    }
                } else if (dst0 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp0;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst1 >= NeuronStart && dst1 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp1);
                    }
                } else if (dst1 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp1;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst2 >= NeuronStart && dst2 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp2);
                    }
                } else if (dst2 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp2;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst3 >= NeuronStart && dst3 < NeuronStart + NeuronTotal) {  
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp3);
                    }
                } else if (dst3 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp3;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst4 >= NeuronStart && dst4 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp4);
                    }
                } else if (dst4 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp4;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst5 >= NeuronStart && dst5 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp5);
                    }
                } else if (dst5 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp5;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst6 >= NeuronStart && dst6 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp6);
                    }
                } else if (dst6 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp6;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
                if (dst7 >= NeuronStart && dst7 < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp7);
                    }
                } else if (dst7 != 0x0) {
                    ap_axiu<64, 0, 0, 0> temp_rt;
                    temp_rt.data = temp7;
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
            }
            
        }
        // Process routed stream from previous router
        ap_axiu<64, 0, 0, 0> temp_rt;
        bool have_rt = SynForwardRoute.read_nb(temp_rt);
        if (have_rt) {
            if (((temp_rt.data >> 32) & 0xFF) == 0xFE) {
                if(coreDone == AmountOfCores - 1) {
                    prev_done = true;
                } else {
                    coreDone++;
                }
            } else {
                ap_uint<24> dst = ((temp_rt.data >> 40) & 0xFFFFFF);
                if (dst >= NeuronStart && dst < NeuronStart + NeuronTotal) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForward.write_nb(temp_rt.data);
                    }
                } else {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_rt);
                    }
                }
            }
        }
    }
    synapse_word_t temp_sync;
    //temp_sync.data = ((ap_uint<64>)0xFFFFFF << 40) | ((ap_uint<64>)0xFE << 32) | ((ap_uint<64>)0x0);
    temp_sync.range(63, 40) = 0xFFFFFF;
    temp_sync.range(39, 32) = 0xFE;
    temp_sync.range(31, 0) = 0x0;
    bool write_status = false;
    while(!write_status) {
        write_status = SynForward.write_nb(temp_sync);
    }
    printf("SynapseRouter done\n");

    // DendriteDelay ------------------------------------------------------------------------------
    printf("DendriteDelay start\n");
    const int DELAY_FIFO_DEPTH = 6000;
    static hls::stream<synapse_word_t> delay_fifo;
    #pragma HLS STREAM variable=delay_fifo depth=DELAY_FIFO_DEPTH
    hls::stream<synapse_word_t> SpikeStream;

    bool done = false;
    int sizeFifo = delay_fifo.size();
    while (!done) {
        #pragma HLS PIPELINE II=1 rewind
        //--------------------------------------------------
        // 1) Age existing packets
        //--------------------------------------------------
        if (sizeFifo > 0) {
            synapse_word_t pkt_in;
            bool read_status = false;
            while(!read_status) {
                read_status = delay_fifo.read_nb(pkt_in);
            }
            if (((pkt_in >> 32) & 0xFF) == 0x0) {
                bool write_status = false;
                while(!write_status) {
                    write_status = SpikeStream.write_nb(pkt_in);
                }
            } else {
                // Decrement & push back
                Delay_t delay = pkt_in.range(39, 32) - 1;
                pkt_in.range(39, 32) = delay;
                bool write_status = false;
                while(!write_status) {
                    write_status = delay_fifo.write_nb(pkt_in);
                }
            }
            sizeFifo--;
        }
        //--------------------------------------------------
        // 2) Accept new packets from SynForward
        //--------------------------------------------------
        synapse_word_t pkt_new;
        bool have_pkt = SynForward.read_nb(pkt_new);
        if (have_pkt) {
            if (((pkt_new >> 40) & 0xFFFFFF) == 0xFFFFFF) {
                // Sync word – forward immediately & exit timestep
                bool write_status = false;
                while(!write_status) {
                    write_status = SpikeStream.write_nb(pkt_new);
                }
                done = true;            // one timestep completed
            } else {
                if (((pkt_new >> 32) & 0xFF) == 0x0) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt_new);
                    }
                } else {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = delay_fifo.write_nb(pkt_new);
                    }
                }
            }
        }
    }
    printf("DendriteDelay done\n");

    // SomaEngine ------------------------------------------------------------------------------
    printf("SomaEngine start\n");
    stream2048u_t spike_status;
    hls::stream<stream2048u_t> SpikeOut_soma;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM_X];
    float I_PreSynCurr[NEURON_NUM_X];
    float R_RefCnt[NEURON_NUM_X];
    float x_state[NEURON_NUM_X];
    float C_acc[NEURON_NUM_X];

    for(int i = 0; i < NEURON_NUM_X; i++) {
        U_membPot[i] = 0;
        I_PreSynCurr[i] = 0;
        R_RefCnt[i] = 0;
        x_state[i] = 0;
        C_acc[i] = 0;
    }

    synapse_loop: while (runstate) {
        //#pragma HLS PIPELINE II=1
        synapse_word_t pkt;
        bool have_pkt = SpikeStream.read_nb(pkt);
        if (have_pkt) {
            float_to_uint32 weight_conv;
            weight_conv.u = pkt.range(31, 0);
            uint32_t dst = pkt.range(63, 40);
            printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt >> 32) & 0xFF), weight_conv.f);
            if (dst == 0xFFFFFF) {
                // Sync – end of timestep
                runstate = false;
            } else {
            // TODO: Demultiplex DstID & compute membrane update
            // *** Insert LIF integration & threshold test ***
                C_acc[dst-NeuronStart] += weight_conv.f;
            }
        }
    }
    for(int i = 0; i < NeuronTotal; i++) {
        I_PreSynCurr[i] = C_acc[i] * w_f;
        printf("C_acc[%d]: %f\n", i, C_acc[i]);
        C_acc[i] = 0;
    }

    spike_status.data = 0;
    printf("SomaEngine Calculate\n");
    for(int i = 0; i < NeuronTotal; i++) {
        x_state[i] = alpha*U_membPot[i] + gamma*I_PreSynCurr[i] + beta*R_RefCnt[i];
        I_PreSynCurr[i] *= beta;
        if(x_state[i] > threshold) {
            spike_status.data |= (ap_uint<2048>)1<<i;
            U_membPot[i] = 0;
            R_RefCnt[i] = t_ref;
        }
        else {
            U_membPot[i] = R_RefCnt[i] > 0 ? 0 : x_state[i];
            R_RefCnt[i] = ((R_RefCnt[i] - 1) > 0) ? (R_RefCnt[i] - 1) : 0;
        }
    }
    printf("SomaEngine Calculate done\n");
    SpikeOut_soma.write(spike_status);
    // SpikeRecorder ------------------------------------------------------------------------------
    stream2048u_t spike_status_soma;
    spike_status_soma = SpikeOut_soma.read();
    for (int i = 0; i < (NeuronTotal/32); i++) {
        SpikeRecorder[i] = (uint32_t)(spike_status_soma.data[i*32] & 0xFFFFFFFF);
    }
    printf("SomaEngine done\n");
    // print size of all streams
    printf("SynapseStream size: %lu\n", SynapseStream.size());
    printf("SynForward size: %lu\n", SynForward.size());
    printf("SynForwardRoute size: %lu\n", SynForwardRoute.size());
    printf("delay_fifo size: %lu\n", delay_fifo.size());
    printf("SpikeStream size: %lu\n", SpikeStream.size());
    printf("SpikeOut_soma size: %lu\n", SpikeOut_soma.size());
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
