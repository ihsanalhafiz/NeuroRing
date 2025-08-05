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
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
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

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM; i++) {
        U_membPot[i] = membrane_potential_float;
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
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            synapse_word_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                float_to_uint32 weight_conv;
                weight_conv.u = pkt.range(31, 0);
                DstID_t dst = pkt.range(63, 40);
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
                } else {
                // TODO: Demultiplex DstID & compute membrane update
                // *** Insert LIF integration & threshold test ***
                    C_acc[dst.to_uint()-NeuronStart] += weight_conv.f;
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
            if(x_state[i] > threshold_float) {
                spike_status.data.range(i, i) = 1;
                U_membPot[i] = membrane_potential_float;
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
    // Pre-compute range bounds for faster comparison
    const uint32_t neuron_end = NeuronStart + NeuronTotal;
    
    router_loop: for (int t = 0; t < SimulationTime; t++) {
        bool axon_done = false;
        bool prev_done = false;
        uint32_t coreDone = 0;
        
        while (!(axon_done && prev_done)) {
        #pragma HLS PIPELINE II=1 rewind
            
            // Process main synapse stream
            stream512u_t pkt;
            bool have_pkt = SynapseStream.read_nb(pkt);
            if (have_pkt) {
                // Extract all 8 synapse entries in parallel
                DstID_t dst[8];
                Delay_t delay[8];
                float_to_uint32 weight_conv[8];
                
                // Unpack all 8 synapses at once
                for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
                    dst[i] = pkt.data.range(base_bit, base_bit - 23);
                    delay[i] = pkt.data.range(base_bit - 24, base_bit - 31);
                    weight_conv[i].u = pkt.data.range(base_bit - 32, base_bit - 63);
                }
                
                printf("weight0: %f, weight1: %f, weight2: %f, weight3: %f, weight4: %f, weight5: %f, weight6: %f, weight7: %f\n", 
                       weight_conv[0].f, weight_conv[1].f, weight_conv[2].f, weight_conv[3].f, 
                       weight_conv[4].f, weight_conv[5].f, weight_conv[6].f, weight_conv[7].f);
                
                // Check if this is an axon done signal
                if (delay[0] == 0xFE) {
                    axon_done = true;
                    ap_axiu<64, 0, 0, 0> temp_sync;
                    temp_sync.data.range(63, 40) = dst[0];
                    temp_sync.data.range(39, 32) = delay[0];
                    temp_sync.data.range(31, 0) = weight_conv[0].u;
                    
                    // Non-blocking write with retry
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(temp_sync);
                    }
                } else {
                    // Process all 8 synapses efficiently
                    synapse_loop: for (int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                        // Create synapse word
                        synapse_word_t temp;
                        temp.range(63, 40) = dst[i];
                        temp.range(39, 32) = delay[i];
                        temp.range(31, 0) = weight_conv[i].u;
                        
                        // Route based on destination
                        bool is_local = (dst[i] >= NeuronStart && dst[i] < neuron_end);
                        bool is_valid = (dst[i] != 0x0);
                        
                        if (is_local) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward.write_nb(temp);
                            }
                        } else if (is_valid) {
                            // Write to routing stream
                            ap_axiu<64, 0, 0, 0> temp_rt;
                            temp_rt.data = temp;
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForwardRoute.write_nb(temp_rt);
                            }
                        }
                    }
                    
                    printf("dst0: %u, dst1: %u, dst2: %u, dst3: %u, dst4: %u, dst5: %u, dst6: %u, dst7: %u\n", 
                           dst[0].to_uint(), dst[1].to_uint(), dst[2].to_uint(), dst[3].to_uint(), 
                           dst[4].to_uint(), dst[5].to_uint(), dst[6].to_uint(), dst[7].to_uint());
                }
            }
            
            // Process routed stream from previous router
            ap_axiu<64, 0, 0, 0> temp_rt;
            bool have_rt = SynapseStreamRoute.read_nb(temp_rt);
            if (have_rt) {
                Delay_t rt_delay = (temp_rt.data >> 32) & 0xFF;
                
                if (rt_delay == 0xFE) {
                    // Handle synchronization
                    if(coreDone == AmountOfCores - 1) {
                        prev_done = true;
                    } else {
                        coreDone++;
                    }
                    
                    // Forward if not for this core
                    ap_uint<24> temp_dst = temp_rt.data.range(63, 40);
                    if (temp_dst != ap_uint<24>(NeuronStart)) {
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForwardRoute.write_nb(temp_rt);
                        }
                    }
                } else {
                    // Route based on destination
                    ap_uint<24> dst = ((temp_rt.data >> 40) & 0xFFFFFF);
                    bool is_local = (dst >= NeuronStart && dst < neuron_end);
                    
                    if (is_local) {
                        // Write to local synapse stream
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForward.write_nb(temp_rt.data);
                        }
                    } else {
                        // Forward to next router
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForwardRoute.write_nb(temp_rt);
                        }
                    }
                }
            }
        }
        
        // Send synchronization signal
        synapse_word_t temp_sync;
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
    const int DELAY_FIFO_DEPTH = 32768;
    //static hls::stream<synapse_word_t> delay_fifo;
    ap_uint<64> buffer_delay[DELAY_FIFO_DEPTH];
    #pragma HLS bind_storage variable=buffer_delay type=ram_2p impl=uram
    uint32_t idx_head = 0;
    uint32_t idx_tail = 0;
    //#pragma HLS STREAM variable=delay_fifo depth=DELAY_FIFO_DEPTH

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        bool done = false;
        int size_buffer = (idx_head >= idx_tail) ? (idx_head - idx_tail) : (idx_head + DELAY_FIFO_DEPTH - idx_tail);
        while (!done) {
            #pragma HLS PIPELINE II=1 rewind
            //--------------------------------------------------
            // 1) Age existing packets
            //--------------------------------------------------
            if (size_buffer > 0) {
                synapse_word_t pkt_in = buffer_delay[idx_tail];
                idx_tail = (idx_tail + 1) % DELAY_FIFO_DEPTH;
                ap_uint<8> delay = pkt_in.range(39, 32);
                if (delay == 0x0) {
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt_in);
                    }
                } else {
                    // Decrement & push back
                    delay = delay - 1;
                    pkt_in.range(39, 32) = delay;
                    buffer_delay[idx_head] = pkt_in;
                    idx_head = (idx_head + 1) % DELAY_FIFO_DEPTH;
                }
                size_buffer--;
            }
            //--------------------------------------------------
            // 2) Accept new packets from SynForward
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward.read_nb(pkt_new);
            if (have_pkt) {
                if (pkt_new.range(63, 40) == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt_new);
                    }
                    done = true;            // one timestep completed
                } else {
                    if (pkt_new.range(39, 32) == 0x0) {
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SpikeStream.write_nb(pkt_new);
                        }
                    } else {
                        buffer_delay[idx_head] = pkt_new;
                        idx_head = (idx_head + 1) % DELAY_FIFO_DEPTH;
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
    uint32_t              threshold,
    uint32_t              membrane_potential,
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
    SomaEngine(
        threshold,
        membrane_potential,
        NeuronStart,
        NeuronTotal,
        spike_stream,
        SimulationTime,
        spike_out
    );
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
