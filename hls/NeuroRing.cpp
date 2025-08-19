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
#define BUF_IDX(core, ofs)   ((core)*DELAY + (ofs))   // ofs == head[core] or (head+delay)

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
    uint32_t                     SimulationTime,
    hls::stream<synapse_word_t>  &SpikeStream,
    hls::stream<ap_uint<128>>  &SpikeOut)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];

    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    ap_uint<128> spike_status;
    spike_status = 0;

    // LIF (iaf_psc_exp) parameters aligned with host_py/network_params.py
    // dt = 0.1 ms, tau_m = 10 ms, tau_syn = 0.5 ms, C_m = 250 pF
    // E_L = -65 mV (leak reversal), V_reset provided via membrane_potential
    const float dt = 0.1f;
    const float tau_m = 10.0f;
    const float tau_syn = 0.5f;
    const float C_m = 250.0f;
    const float E_L = -65.0f;
    const float V_decay = 0.99004983f;   // exp(-dt/tau_m)
    const float I_decay = 0.81873075f;   // exp(-dt/tau_syn)
    const float syn_to_vm = (1.0f/C_m) * ((I_decay - V_decay) / ((1.0f/tau_m) - (1.0f/tau_syn)));
    const int   t_ref_steps = 20;        // round(2.0/0.1)

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];   // membrane potential (mV)
    float I_PreSynCurr[NEURON_NUM/8];
    uint16_t R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        U_membPot[core] = membrane_potential_float;
        I_PreSynCurr[core] = 0;
        R_RefCnt[core] = 0;
        x_state[core] = 0;
        C_acc[core] = 0;
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }
    
    //----------------------------------------------------------
    // Timestep loop
    //----------------------------------------------------------
    timestep_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) Broadcast spikes from last step to AxonLoader
        //--------------------------------------------------
        spike_status = 0;

        for(int i = 0; i < NeuronTotal; i++) {
            // Refractory countdown
            if (R_RefCnt[i] > 0) {
                R_RefCnt[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot[i];
                float i_prev = I_PreSynCurr[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot[i] = v_new;
                // Spike generation
                if (U_membPot[i] >= threshold_float) {
                    spike_status.range(i, i) = 1;
                    U_membPot[i] = membrane_potential_float;
                    R_RefCnt[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        bool write_status = false;
        while(!write_status) {
            write_status = SpikeOut.write_nb(spike_status);
        }
    

        bool done = false;
        while (!done) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SpikeStream.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                ap_uint<6> h2 = (head[dst.to_int()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_int()-NeuronStart),h2)];
                if (delay == 0xFC) {
                    U_membPot[dst.to_uint()-NeuronStart] = weight;
                }
                else if (delay == 0xFE) {
                    done = true;
                } else {
                    buf_flat[BUF_IDX((dst.to_int()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NeuronTotal; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head[i]) & 0x3F;
            I_PreSynCurr[i] = I_PreSynCurr[i] * I_decay + buf_flat[BUF_IDX(i,h3)];
            buf_flat[BUF_IDX(i,h3)] = 0;
            head[i] = (h3 + 1) & 0x3F;
        }
    }
}

void accumulate_spike(
    int SimulationTime,
    hls::stream<ap_uint<128>> &spike_out,
    hls::stream<ap_uint<128>> &spike_out1,
    hls::stream<ap_uint<128>> &spike_out2,
    hls::stream<ap_uint<128>> &spike_out3,
    hls::stream<ap_uint<128>> &spike_out4,
    hls::stream<ap_uint<128>> &spike_out5,
    hls::stream<ap_uint<128>> &spike_out6,
    hls::stream<ap_uint<128>> &spike_out7,
    hls::stream<ap_uint<128>> &spike_out8,
    hls::stream<ap_uint<128>> &spike_out9,
    hls::stream<ap_uint<128>> &spike_out10,
    hls::stream<ap_uint<128>> &spike_out11,
    hls::stream<ap_uint<128>> &spike_out12,
    hls::stream<ap_uint<128>> &spike_out13,
    hls::stream<ap_uint<128>> &spike_out14,
    hls::stream<ap_uint<128>> &spike_out15,
    hls::stream<stream2048u_t> &spike_stream
)
{
    stream2048u_t pkt_spike;
    for (int t = 0; t < SimulationTime; t++) {
        pkt_spike.data = 0;
        ap_uint<128> spike_status;
        bool read_status = false;
        while(!read_status) {
            read_status = spike_out.read_nb(spike_status);
        }
        pkt_spike.data.range(127, 0) = spike_status;
        bool read_status1 = false;
        while(!read_status1) {
            read_status1 = spike_out1.read_nb(spike_status);
        }
        pkt_spike.data.range(255, 128) = spike_status;
        bool read_status2 = false;
        while(!read_status2) {
            read_status2 = spike_out2.read_nb(spike_status);
        }
        pkt_spike.data.range(383, 256) = spike_status;
        bool read_status3 = false;
        while(!read_status3) {
            read_status3 = spike_out3.read_nb(spike_status);
        }
        pkt_spike.data.range(511, 384) = spike_status;
        bool read_status4 = false;
        while(!read_status4) {
            read_status4 = spike_out4.read_nb(spike_status);
        }
        pkt_spike.data.range(639, 512) = spike_status;
        bool read_status5 = false;
        while(!read_status5) {
            read_status5 = spike_out5.read_nb(spike_status);
        }
        pkt_spike.data.range(767, 640) = spike_status;
        bool read_status6 = false;
        while(!read_status6) {
            read_status6 = spike_out6.read_nb(spike_status);
        }
        pkt_spike.data.range(895, 768) = spike_status;
        bool read_status7 = false;
        while(!read_status7) {
            read_status7 = spike_out7.read_nb(spike_status);
        }
        pkt_spike.data.range(1023, 896) = spike_status;  
        bool read_status8 = false;
        while(!read_status8) {
            read_status8 = spike_out8.read_nb(spike_status);
        }
        pkt_spike.data.range(1151, 1024) = spike_status;
        bool read_status9 = false;
        while(!read_status9) {
            read_status9 = spike_out9.read_nb(spike_status);
        }
        pkt_spike.data.range(1279, 1152) = spike_status;
        bool read_status10 = false;
        while(!read_status10) {
            read_status10 = spike_out10.read_nb(spike_status);
        }
        pkt_spike.data.range(1407, 1280) = spike_status;
        bool read_status11 = false;
        while(!read_status11) {
            read_status11 = spike_out11.read_nb(spike_status);
        }
        pkt_spike.data.range(1535, 1408) = spike_status;
        bool read_status12 = false;
        while(!read_status12) {
            read_status12 = spike_out12.read_nb(spike_status);
        }
        pkt_spike.data.range(1663, 1536) = spike_status;
        bool read_status13 = false;
        while(!read_status13) {
            read_status13 = spike_out13.read_nb(spike_status);
        }
        pkt_spike.data.range(1791, 1664) = spike_status;
        bool read_status14 = false;
        while(!read_status14) {
            read_status14 = spike_out14.read_nb(spike_status);
        }
        pkt_spike.data.range(1919, 1792) = spike_status;
        bool read_status15 = false;
        while(!read_status15) {
            read_status15 = spike_out15.read_nb(spike_status);
        }
        pkt_spike.data.range(2047, 1920) = spike_status;

        bool write_status = false;
        while(!write_status) {
            write_status = spike_stream.write_nb(pkt_spike);
        }
    }
}


//====================================================================
//  3. SynapseRouter – Route packets to local or next core
//====================================================================
void SynapseRouter(
    hls::stream<stream1024u_t> &SynapseStream,
    hls::stream<stream1024u_t> &SynapseStreamRoute,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     SimulationTime,
    uint32_t                     AmountOfCores,
    hls::stream<synapse_word_t> &SynForward,
    hls::stream<synapse_word_t> &SynForward1,
    hls::stream<synapse_word_t> &SynForward2,
    hls::stream<synapse_word_t> &SynForward3,
    hls::stream<synapse_word_t> &SynForward4,
    hls::stream<synapse_word_t> &SynForward5,
    hls::stream<synapse_word_t> &SynForward6,
    hls::stream<synapse_word_t> &SynForward7,
    hls::stream<synapse_word_t> &SynForward8,
    hls::stream<synapse_word_t> &SynForward9,
    hls::stream<synapse_word_t> &SynForward10,
    hls::stream<synapse_word_t> &SynForward11,
    hls::stream<synapse_word_t> &SynForward12,
    hls::stream<synapse_word_t> &SynForward13,
    hls::stream<synapse_word_t> &SynForward14,
    hls::stream<synapse_word_t> &SynForward15,
    hls::stream<stream1024u_t> &SynForwardRoute)
{
    // Pre-compute range bounds for faster comparison
    const uint32_t neuron_end = NeuronStart + NeuronTotal;
    ap_uint<24> start[16];
    ap_uint<24> end[16];
    #pragma HLS ARRAY_PARTITION variable=start complete
    #pragma HLS ARRAY_PARTITION variable=end complete

    start[0] = NeuronStart;
    start[1] = NeuronStart + ((NeuronTotal+15)/16)*1;
    start[2] = NeuronStart + ((NeuronTotal+15)/16)*2;
    start[3] = NeuronStart + ((NeuronTotal+15)/16)*3;
    start[4] = NeuronStart + ((NeuronTotal+15)/16)*4;
    start[5] = NeuronStart + ((NeuronTotal+15)/16)*5;
    start[6] = NeuronStart + ((NeuronTotal+15)/16)*6;
    start[7] = NeuronStart + ((NeuronTotal+15)/16)*7;
    start[8] = NeuronStart + ((NeuronTotal+15)/16)*8;
    start[9] = NeuronStart + ((NeuronTotal+15)/16)*9;
    start[10] = NeuronStart + ((NeuronTotal+15)/16)*10;
    start[11] = NeuronStart + ((NeuronTotal+15)/16)*11;
    start[12] = NeuronStart + ((NeuronTotal+15)/16)*12;
    start[13] = NeuronStart + ((NeuronTotal+15)/16)*13;
    start[14] = NeuronStart + ((NeuronTotal+15)/16)*14;
    start[15] = NeuronStart + ((NeuronTotal+15)/16)*15;

    end[0] = start[1];
    end[1] = start[2];
    end[2] = start[3];
    end[3] = start[4];
    end[4] = start[5];
    end[5] = start[6];
    end[6] = start[7];
    end[7] = start[8];
    end[8] = start[9];
    end[9] = start[10];
    end[10] = start[11];
    end[11] = start[12];
    end[12] = start[13];
    end[13] = start[14];
    end[14] = start[15];
    end[15] = neuron_end;

    bool read_axonLoader = true;
    hls::vector<DstID_t, 16> dst_forward = (DstID_t)0;
    hls::vector<Delay_t, 16> delay_forward = (Delay_t)0;
    hls::vector<uint32_t, 16> weight_bits_forward = (uint32_t)0;
    #pragma HLS ARRAY_PARTITION variable=dst_forward complete
    #pragma HLS ARRAY_PARTITION variable=delay_forward complete
    #pragma HLS ARRAY_PARTITION variable=weight_bits_forward complete
    int size_forward = 0;
    
    router_loop: for (int t = 0; t < SimulationTime; t++) {
        bool axon_done = false;
        bool prev_done = false;
        uint32_t coreDone = 0;
        size_forward = 0;
        
        while (!(axon_done && prev_done)) {
        #pragma HLS PIPELINE II=1 rewind
            
            // Process main synapse stream
            stream1024u_t pkt;
            bool have_pkt = false;
            if(read_axonLoader) {
                have_pkt = SynapseStream.read_nb(pkt);
                if(!have_pkt) {
                    have_pkt = SynapseStreamRoute.read_nb(pkt);
                } else {
                    read_axonLoader = false;
                }
            } else {
                have_pkt = SynapseStreamRoute.read_nb(pkt);
                if(!have_pkt) {
                    have_pkt = SynapseStream.read_nb(pkt);
                } else {
                    read_axonLoader = true;
                }   
            }
            if (have_pkt) {
                // Extract all 16 synapse entries in parallel
                DstID_t dst[16];
                Delay_t delay[16];
                uint32_t weight_bits[16];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=delay complete
                #pragma HLS ARRAY_PARTITION variable=weight_bits complete
                
                // Unpack all 16 synapses at once
                for (int i = 0; i < 16; i++) {
                #pragma HLS UNROLL
                    int base_bit = 1023 - i * 64;
                    dst[i] = pkt.data.range(base_bit, base_bit - 23);
                    delay[i] = pkt.data.range(base_bit - 24, base_bit - 31);
                    weight_bits[i] = pkt.data.range(base_bit - 32, base_bit - 63);
                }

                synapse_loop: for (int i = 0; i < 16; i++) {
                    //#pragma HLS UNROLL
                    // Create synapse word
                    synapse_word_t temp;
                    temp.range(63, 40) = dst[i];
                    temp.range(39, 32) = delay[i];
                    temp.range(31, 0)  = weight_bits[i];

                    // Find region index in parallel
                    int region = -1;
                    find_region: for (int r = 0; r < 16; ++r) {
                        #pragma HLS UNROLL
                        bool is_local = (dst[i] >= start[r] && dst[i] < end[r]);
                        if (is_local && region == -1) region = r;
                    }

                    // Dispatch to the right stream (prefer blocking write to avoid spin-wait)
                    switch (region) {
                        case 0:  SynForward.write(temp); break;
                        case 1:  SynForward1.write(temp); break;
                        case 2:  SynForward2.write(temp); break;
                        case 3:  SynForward3.write(temp); break;
                        case 4:  SynForward4.write(temp); break;
                        case 5:  SynForward5.write(temp); break;
                        case 6:  SynForward6.write(temp); break;
                        case 7:  SynForward7.write(temp); break;
                        case 8:  SynForward8.write(temp); break;
                        case 9:  SynForward9.write(temp); break;
                        case 10: SynForward10.write(temp); break;
                        case 11: SynForward11.write(temp); break;
                        case 12: SynForward12.write(temp); break;
                        case 13: SynForward13.write(temp); break;
                        case 14: SynForward14.write(temp); break;
                        case 15: SynForward15.write(temp); break;
                        default: break; // no-op if no region matched
                    }
                    if(region != -1) {
                        // Clear once
                        dst[i] = 0;
                        delay[i] = 0;
                        weight_bits[i] = 0;
                    }
                }                
                // Check if this is an axon done signal
                if (delay[0] == 0xFE) {
                    if(dst[0] == NeuronStart) {
                        axon_done = true;
                        // Non-blocking write with retry
                        if(size_forward > 0) {
                            stream1024u_t pkt_forward;
                            for(int i = 0; i < 16; i++) {
                                #pragma HLS UNROLL
                                int base_bit = 1023 - i * 64;
                                pkt_forward.data.range(base_bit, base_bit - 23) = dst_forward[i];
                                pkt_forward.data.range(base_bit - 24, base_bit - 31) = delay_forward[i];
                                pkt_forward.data.range(base_bit - 32, base_bit - 63) = weight_bits_forward[i];
                                dst_forward[i] = (DstID_t)0;
                                delay_forward[i] = (Delay_t)0;
                                weight_bits_forward[i] = (uint32_t)0;
                            }
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForwardRoute.write_nb(pkt_forward);
                            }
                            size_forward = 0;
                        }
                    }

                    if(coreDone == AmountOfCores) {
                        prev_done = true;
                        if(dst[0] != NeuronStart) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForwardRoute.write_nb(pkt);
                            }
                        }
                    } else {
                        coreDone++;
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForwardRoute.write_nb(pkt);
                        }
                    }
                } else {
                    for(int i = 0; i < 16; i++) {
                        if(size_forward < 16) {
                            dst_forward[size_forward] = dst[i];
                            delay_forward[size_forward] = delay[i];
                            weight_bits_forward[size_forward] = weight_bits[i];
                            size_forward++;
                        } else {
                            stream1024u_t pkt_forward;
                            for(int i = 0; i < 16; i++) {
                                int base_bit = 1023 - i * 64;
                                pkt_forward.data.range(base_bit, base_bit - 23) = dst_forward[i];
                                pkt_forward.data.range(base_bit - 24, base_bit - 31) = delay_forward[i];
                                pkt_forward.data.range(base_bit - 32, base_bit - 63) = weight_bits_forward[i];
                                dst_forward[i] = (DstID_t)0;
                                delay_forward[i] = (Delay_t)0;
                                weight_bits_forward[i] = (uint32_t)0;
                            }
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForwardRoute.write_nb(pkt_forward);
                            }
                            size_forward = 0;
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
        bool write_status1 = false;
        while(!write_status1) {
            write_status1 = SynForward1.write_nb(temp_sync);
        }
        bool write_status2 = false;
        while(!write_status2) {
            write_status2 = SynForward2.write_nb(temp_sync);
        }
        bool write_status3 = false;
        while(!write_status3) {
            write_status3 = SynForward3.write_nb(temp_sync);
        }
        bool write_status4 = false;
        while(!write_status4) {
            write_status4 = SynForward4.write_nb(temp_sync);
        }
        bool write_status5 = false;
        while(!write_status5) {
            write_status5 = SynForward5.write_nb(temp_sync);
        }
        bool write_status6 = false;
        while(!write_status6) {
            write_status6 = SynForward6.write_nb(temp_sync);
        }
        bool write_status7 = false;
        while(!write_status7) {
            write_status7 = SynForward7.write_nb(temp_sync);
        }
        bool write_status8 = false;
        while(!write_status8) {
            write_status8 = SynForward8.write_nb(temp_sync);
        }
        bool write_status9 = false;
        while(!write_status9) {
            write_status9 = SynForward9.write_nb(temp_sync);
        }   
        bool write_status10 = false;
        while(!write_status10) {
            write_status10 = SynForward10.write_nb(temp_sync);
        }
        bool write_status11 = false;
        while(!write_status11) {
            write_status11 = SynForward11.write_nb(temp_sync);
        }
        bool write_status12 = false;
        while(!write_status12) {
            write_status12 = SynForward12.write_nb(temp_sync);
        }
        bool write_status13 = false;
        while(!write_status13) {
            write_status13 = SynForward13.write_nb(temp_sync);
        }
        bool write_status14 = false;
        while(!write_status14) {
            write_status14 = SynForward14.write_nb(temp_sync);
        }
        bool write_status15 = false;
        while(!write_status15) {
            write_status15 = SynForward15.write_nb(temp_sync);
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
    hls::stream<stream1024u_t> &syn_route_in,
    hls::stream<stream1024u_t> &syn_forward_rt,
    hls::stream<stream1024u_t> &synapse_stream,
    hls::stream<stream2048u_t> &spike_out_axon)
{
#pragma HLS INTERFACE axis port=synapse_stream bundle=AXIS_IN
#pragma HLS INTERFACE axis port=spike_out_axon bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=syn_route_in bundle=AXIS_IN
#pragma HLS INTERFACE axis port=syn_forward_rt bundle=AXIS_OUT

//---------------------------
//  On‑chip FIFO channels
//---------------------------
#pragma HLS DATAFLOW

    hls::stream<synapse_word_t> spike_stream;
    hls::stream<synapse_word_t> spike_stream1;
    hls::stream<synapse_word_t> spike_stream2;
    hls::stream<synapse_word_t> spike_stream3;
    hls::stream<synapse_word_t> spike_stream4;
    hls::stream<synapse_word_t> spike_stream5;
    hls::stream<synapse_word_t> spike_stream6;
    hls::stream<synapse_word_t> spike_stream7;
    hls::stream<synapse_word_t> spike_stream8;
    hls::stream<synapse_word_t> spike_stream9;
    hls::stream<synapse_word_t> spike_stream10;
    hls::stream<synapse_word_t> spike_stream11;
    hls::stream<synapse_word_t> spike_stream12;
    hls::stream<synapse_word_t> spike_stream13;
    hls::stream<synapse_word_t> spike_stream14;
    hls::stream<synapse_word_t> spike_stream15;

    hls::stream<ap_uint<128>> spike_out;
    hls::stream<ap_uint<128>> spike_out1;
    hls::stream<ap_uint<128>> spike_out2;
    hls::stream<ap_uint<128>> spike_out3;
    hls::stream<ap_uint<128>> spike_out4;
    hls::stream<ap_uint<128>> spike_out5;
    hls::stream<ap_uint<128>> spike_out6;
    hls::stream<ap_uint<128>> spike_out7;
    hls::stream<ap_uint<128>> spike_out8;
    hls::stream<ap_uint<128>> spike_out9;
    hls::stream<ap_uint<128>> spike_out10;
    hls::stream<ap_uint<128>> spike_out11;
    hls::stream<ap_uint<128>> spike_out12;
    hls::stream<ap_uint<128>> spike_out13;
    hls::stream<ap_uint<128>> spike_out14;
    hls::stream<ap_uint<128>> spike_out15;

#pragma HLS STREAM variable=spike_stream    depth=256
#pragma HLS STREAM variable=spike_stream1    depth=256
#pragma HLS STREAM variable=spike_stream2    depth=256
#pragma HLS STREAM variable=spike_stream3    depth=256
#pragma HLS STREAM variable=spike_stream4    depth=256
#pragma HLS STREAM variable=spike_stream5    depth=256
#pragma HLS STREAM variable=spike_stream6    depth=256
#pragma HLS STREAM variable=spike_stream7    depth=256
#pragma HLS STREAM variable=spike_stream8    depth=256
#pragma HLS STREAM variable=spike_stream9    depth=256
#pragma HLS STREAM variable=spike_stream10   depth=256
#pragma HLS STREAM variable=spike_stream11   depth=256
#pragma HLS STREAM variable=spike_stream12   depth=256
#pragma HLS STREAM variable=spike_stream13   depth=256
#pragma HLS STREAM variable=spike_stream14   depth=256
#pragma HLS STREAM variable=spike_stream15   depth=256

#pragma HLS STREAM variable=spike_out    depth=32
#pragma HLS STREAM variable=spike_out1    depth=32
#pragma HLS STREAM variable=spike_out2    depth=32
#pragma HLS STREAM variable=spike_out3    depth=32
#pragma HLS STREAM variable=spike_out4    depth=32
#pragma HLS STREAM variable=spike_out5    depth=32
#pragma HLS STREAM variable=spike_out6    depth=32
#pragma HLS STREAM variable=spike_out7    depth=32
#pragma HLS STREAM variable=spike_out8    depth=32
#pragma HLS STREAM variable=spike_out9    depth=32
#pragma HLS STREAM variable=spike_out10   depth=32
#pragma HLS STREAM variable=spike_out11   depth=32
#pragma HLS STREAM variable=spike_out12   depth=32
#pragma HLS STREAM variable=spike_out13   depth=32
#pragma HLS STREAM variable=spike_out14   depth=32
#pragma HLS STREAM variable=spike_out15   depth=32

    // Launch data‑flow processes
    // Note: You may need to define a threshold value for SomaEngine, e.g., params.threshold if available
    SynapseRouter(
        synapse_stream,
        syn_route_in,
        NeuronStart,
        NeuronTotal,
        SimulationTime,
        AmountOfCores,
        spike_stream,
        spike_stream1,
        spike_stream2,
        spike_stream3,
        spike_stream4,
        spike_stream5,
        spike_stream6,
        spike_stream7,
        spike_stream8,
        spike_stream9,
        spike_stream10,
        spike_stream11,
        spike_stream12,
        spike_stream13,
        spike_stream14,
        spike_stream15,
        syn_forward_rt
    );

    SE0: SomaEngine(threshold, membrane_potential, 
        NeuronStart, ((NeuronTotal+15)/16), SimulationTime, spike_stream, spike_out);

    SE1: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*1), ((NeuronTotal+15)/16), SimulationTime, spike_stream1, 
        spike_out1
    );
    SE2: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*2), ((NeuronTotal+15)/16), SimulationTime, spike_stream2, 
        spike_out2
    );
    SE3: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*3), ((NeuronTotal+15)/16), SimulationTime, spike_stream3, 
        spike_out3
    );
    SE4: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*4), ((NeuronTotal+15)/16), SimulationTime, spike_stream4, 
        spike_out4
    );
    SE5: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*5), ((NeuronTotal+15)/16), SimulationTime, spike_stream5, 
        spike_out5
    );
    SE6: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*6), ((NeuronTotal+15)/16), SimulationTime, spike_stream6, 
        spike_out6
    );
    SE7: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*7), ((NeuronTotal+15)/16), SimulationTime, spike_stream7, 
        spike_out7
    );
    SE8: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*8), ((NeuronTotal+15)/16), SimulationTime, spike_stream8, 
        spike_out8
    );
    SE9: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*9), ((NeuronTotal+15)/16), SimulationTime, spike_stream9, 
        spike_out9
    );
    SE10: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*10), ((NeuronTotal+15)/16), SimulationTime, spike_stream10, 
        spike_out10
    );
    SE11: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*11), ((NeuronTotal+15)/16), SimulationTime, spike_stream11, 
        spike_out11
    );
    SE12: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*12), ((NeuronTotal+15)/16), SimulationTime, spike_stream12, 
        spike_out12
    );
    SE13: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*13), ((NeuronTotal+15)/16), SimulationTime, spike_stream13, 
        spike_out13
    );
    SE14: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*14), ((NeuronTotal+15)/16), SimulationTime, spike_stream14, 
        spike_out14
    );
    SE15: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+15)/16)*15), NeuronTotal - (((NeuronTotal+15)/16)*15), SimulationTime, spike_stream15, 
        spike_out15
    );

    accumulate_spike(SimulationTime, spike_out, spike_out1, spike_out2, spike_out3, spike_out4, spike_out5, spike_out6, spike_out7, spike_out8, spike_out9, spike_out10, spike_out11, spike_out12, spike_out13, spike_out14, spike_out15, spike_out_axon);
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
