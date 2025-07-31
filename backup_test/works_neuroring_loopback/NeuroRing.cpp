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

#define NEURON_NUM 2048
#define SYNAPSE_LIST_SIZE 5000

//====================================================================
//  Parameter & helper structs ‑ extend as required
//====================================================================
// (All typedefs, structs, and function prototypes removed, as they are now in NeuroRing.h)

//====================================================================
//  1. SomaEngine – Computes neuron dynamics & produces spike bitmap
//====================================================================
void SomaEngine(
    float                        threshold,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<synapse_word_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream2048u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream2048u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM];
    float I_PreSynCurr[NEURON_NUM];
    float R_RefCnt[NEURON_NUM];
    float x_state[NEURON_NUM];
    float C_acc[NEURON_NUM];

    for(int i = 0; i < NEURON_NUM; i++) {
        U_membPot[i] = 0;
        I_PreSynCurr[i] = 0;
        R_RefCnt[i] = 0;
        x_state[i] = 0;
        C_acc[i] = 0;
    }

    //----------------------------------------------------------
    // Timestep loop
    //----------------------------------------------------------
    timestep_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) Broadcast spikes from last step to AxonLoader
        //--------------------------------------------------
        
        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            synapse_word_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                float_to_uint32 weight_conv;
                weight_conv.u = pkt.range(31, 0);
                uint32_t dst = pkt.range(63, 40);
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
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
        
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NeuronTotal; i++) {
            I_PreSynCurr[i] += C_acc[i] * w_f;
            C_acc[i] = 0;
        }

        spike_status.data = 0;

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
        bool write_status = false;
        while(!write_status) {
            write_status = SpikeOut.write_nb(spike_status);
        }


    }
}


//====================================================================
//  3. SynapseRouter – Route packets to local or next core
//====================================================================
void SynapseRouter(
    hls::stream<stream512u_t> &SynapseStream,
    hls::stream<ap_axiu<64, 0, 0, 0>> &SynapseStreamRoute,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     SimulationTime,
    uint32_t                     AmountOfCores,
    hls::stream<synapse_word_t> &SynForward,
    hls::stream<ap_axiu<64, 0, 0, 0>> &SynForwardRoute)
{
    router_loop: for (int t = 0; t < SimulationTime; t++) {
        bool axon_done = false;
        bool prev_done = false;
        uint32_t coreDone = 0;
        while (!(axon_done && prev_done)) {
        #pragma HLS PIPELINE II=1 rewind
            stream512u_t pkt;
            bool have_pkt = SynapseStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst0 = pkt.data.range(511, 488);
                Delay_t delay0 = pkt.data.range(487, 480);
                float_to_uint32 weight_conv0;
                weight_conv0.u = pkt.data.range(479, 448);
    
                DstID_t dst1 = pkt.data.range(447, 424);
                Delay_t delay1 = pkt.data.range(423, 416);
                float_to_uint32 weight_conv1;
                weight_conv1.u = pkt.data.range(415, 384);
    
                DstID_t dst2 = pkt.data.range(383, 360);
                Delay_t delay2 = pkt.data.range(359, 352);
                float_to_uint32 weight_conv2;
                weight_conv2.u = pkt.data.range(351, 320);
    
                DstID_t dst3 = pkt.data.range(319, 296);
                Delay_t delay3 = pkt.data.range(295, 288);
                float_to_uint32 weight_conv3;
                weight_conv3.u = pkt.data.range(287, 256);
                
                DstID_t dst4 = pkt.data.range(255, 232);
                Delay_t delay4 = pkt.data.range(231, 224);
                float_to_uint32 weight_conv4;
                weight_conv4.u = pkt.data.range(223, 192);
    
                DstID_t dst5 = pkt.data.range(191, 168);
                Delay_t delay5 = pkt.data.range(167, 160);
                float_to_uint32 weight_conv5;
                weight_conv5.u = pkt.data.range(159, 128);
    
                DstID_t dst6 = pkt.data.range(127, 104);
                Delay_t delay6 = pkt.data.range(103, 96);
                float_to_uint32 weight_conv6;
                weight_conv6.u = pkt.data.range(95, 64);
    
                DstID_t dst7 = pkt.data.range(63, 40);
                Delay_t delay7 = pkt.data.range(39, 32);
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
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_sync);
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
                            write_status =  SynForward.write_nb(temp3);
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
            }    // Process routed stream from previous router
            ap_axiu<64, 0, 0, 0> temp_rt;
            bool have_rt = SynapseStreamRoute.read_nb(temp_rt);
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
    }
}

//====================================================================
//  4. DendriteDelay – Implements discrete delay line per synapse
//====================================================================
void DendriteDelay(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     SimulationTime,
    hls::stream<synapse_word_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    // 6000‑deep FIFO chosen from spec; tweak as needed.
    const int DELAY_FIFO_DEPTH = 6000;
    static hls::stream<synapse_word_t> delay_fifo;
    #pragma HLS STREAM variable=delay_fifo depth=DELAY_FIFO_DEPTH

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
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
    }
}

//--------------------------------------------------------------------
//  Top‑level kernel ‒ integrates all sub‑kernels using DATAFLOW
//--------------------------------------------------------------------
extern "C" void NeuroRing(
    uint32_t              SimulationTime,
    float                 threshold,
    uint32_t              AmountOfCores,
    uint32_t              NeuronStart,
    uint32_t              NeuronTotal,
    hls::stream<ap_axiu<64, 0, 0, 0>> &syn_route_in,
    hls::stream<ap_axiu<64, 0, 0, 0>> &syn_forward_rt,
    hls::stream<stream512u_t> &synapse_stream,
    hls::stream<stream2048u_t> &spike_out
)
{
#pragma HLS INTERFACE axis port=synapse_stream bundle=AXIS_IN
#pragma HLS INTERFACE axis port=spike_out bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=syn_route_in bundle=AXIS_IN
#pragma HLS INTERFACE axis port=syn_forward_rt bundle=AXIS_OUT


//---------------------------
//  On‑chip FIFO channels
//---------------------------
#pragma HLS DATAFLOW
    hls::stream<synapse_word_t> syn_forward;
    hls::stream<synapse_word_t> spike_stream;

#pragma HLS STREAM variable=syn_forward     depth=1024
#pragma HLS STREAM variable=spike_stream    depth=1024

    // Launch data‑flow processes
    // Note: You may need to define a threshold value for SomaEngine, e.g., params.threshold if available
    SomaEngine(
        threshold,
        NeuronStart,
        NeuronTotal,
        spike_stream,
        SimulationTime,
        spike_out
    );
    SynapseRouter(
        synapse_stream,
        syn_route_in,
        NeuronStart,
        NeuronTotal,
        SimulationTime,
        AmountOfCores,
        syn_forward,
        syn_forward_rt
    );
    DendriteDelay(
        syn_forward,
        SimulationTime,
        spike_stream
    );
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
