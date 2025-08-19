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
    hls::stream<ap_uint<256>>  &SpikeOut)
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
    ap_uint<256> spike_status;
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

    float U_membPot[NCORE];   // membrane potential (mV)
    float I_PreSynCurr[NCORE];
    uint16_t R_RefCnt[NCORE];
    float x_state[NCORE];
    float C_acc[NCORE];

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
    hls::stream<ap_uint<256>> &spike_out,
    hls::stream<ap_uint<256>> &spike_out1,
    hls::stream<ap_uint<256>> &spike_out2,
    hls::stream<ap_uint<256>> &spike_out3,
    hls::stream<ap_uint<256>> &spike_out4,
    hls::stream<ap_uint<256>> &spike_out5,
    hls::stream<ap_uint<256>> &spike_out6,
    hls::stream<ap_uint<256>> &spike_out7,
    hls::stream<stream2048u_t> &spike_stream
)
{
    stream2048u_t pkt_spike;
    for (int t = 0; t < SimulationTime; t++) {
        pkt_spike.data = 0;
        ap_uint<256> spike_status;
        bool read_status = false;
        while(!read_status) {
            read_status = spike_out.read_nb(spike_status);
        }
        pkt_spike.data.range(255, 0) = spike_status;
        bool read_status1 = false;
        while(!read_status1) {
            read_status1 = spike_out1.read_nb(spike_status);
        }
        pkt_spike.data.range(511, 256) = spike_status;
        bool read_status2 = false;
        while(!read_status2) {
            read_status2 = spike_out2.read_nb(spike_status);
        }
        pkt_spike.data.range(767, 512) = spike_status;
        bool read_status3 = false;
        while(!read_status3) {
            read_status3 = spike_out3.read_nb(spike_status);
        }
        pkt_spike.data.range(1023, 768) = spike_status;
        bool read_status4 = false;
        while(!read_status4) {
            read_status4 = spike_out4.read_nb(spike_status);
        }
        pkt_spike.data.range(1279, 1024) = spike_status;
        bool read_status5 = false;
        while(!read_status5) {
            read_status5 = spike_out5.read_nb(spike_status);
        }
        pkt_spike.data.range(1535, 1280) = spike_status;
        bool read_status6 = false;
        while(!read_status6) {
            read_status6 = spike_out6.read_nb(spike_status);
        }
        pkt_spike.data.range(1791, 1536) = spike_status;
        bool read_status7 = false;
        while(!read_status7) {
            read_status7 = spike_out7.read_nb(spike_status);
        }
        pkt_spike.data.range(2047, 1792) = spike_status;  

        bool write_ok = false;
        while(!write_ok) {
            write_ok = spike_stream.write_nb(pkt_spike);
        }
    }
}


//====================================================================
//  3. SynapseRouter – Route packets to local or next core
//====================================================================
void SynapseRouter(
    hls::stream<stream512u_t> &SynapseStream,
    hls::stream<stream512u_t> &SynapseStreamRoute,
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
    hls::stream<stream512u_t> &SynForwardRoute)
{
    // Pre-compute range bounds for faster comparison
    const uint32_t neuron_end = NeuronStart + NeuronTotal;
    ap_uint<24> start[8];
    ap_uint<24> end[8];
    #pragma HLS ARRAY_PARTITION variable=start complete
    #pragma HLS ARRAY_PARTITION variable=end complete

    start[0] = NeuronStart;
    start[1] = NeuronStart + ((NeuronTotal+7)/8)*1;
    start[2] = NeuronStart + ((NeuronTotal+7)/8)*2;
    start[3] = NeuronStart + ((NeuronTotal+7)/8)*3;
    start[4] = NeuronStart + ((NeuronTotal+7)/8)*4;
    start[5] = NeuronStart + ((NeuronTotal+7)/8)*5;
    start[6] = NeuronStart + ((NeuronTotal+7)/8)*6;
    start[7] = NeuronStart + ((NeuronTotal+7)/8)*7;

    end[0] = start[1];
    end[1] = start[2];
    end[2] = start[3];
    end[3] = start[4];
    end[4] = start[5];
    end[5] = start[6];
    end[6] = start[7];
    end[7] = neuron_end;

    bool read_axonLoader = true;
    // Forward packet aggregation buffers (pack 8 synapses → one 512-bit word)
    DstID_t   dst_forward[8];
    Delay_t   delay_forward[8];
    uint32_t  weight_bits_forward[8];
    #pragma HLS ARRAY_PARTITION variable=dst_forward complete
    #pragma HLS ARRAY_PARTITION variable=delay_forward complete
    #pragma HLS ARRAY_PARTITION variable=weight_bits_forward complete
    ap_uint<4> size_forward = 0; // 0..8

    // Non-blocking pending writes to avoid spin-wait inside the pipeline
    bool forward_pending = false;
    stream512u_t pending_forward_pkt;
    bool route_pending = false;
    stream512u_t pending_route_pkt;
    
    router_loop: for (int t = 0; t < SimulationTime; t++) {
        bool axon_done = false;
        bool prev_done = false;
        uint32_t coreDone = 0;
        size_forward = 0;
        
        while (!(axon_done && prev_done)) {
        #pragma HLS PIPELINE II=1 rewind
            // 1) First, try to drain any pending non-blocking writes
            if (forward_pending) {
                bool wrote_fw = SynForwardRoute.write_nb(pending_forward_pkt);
                if (wrote_fw) {
                    forward_pending = false;
                    // Clear aggregation buffers only after successful transmit
                    for (int i = 0; i < 8; i++) {
                        #pragma HLS UNROLL
                        dst_forward[i] = (DstID_t)0;
                        delay_forward[i] = (Delay_t)0;
                        weight_bits_forward[i] = (uint32_t)0;
                    }
                    size_forward = 0;
                }
            }
            if (!forward_pending && route_pending) {
                bool wrote_rt = SynForwardRoute.write_nb(pending_route_pkt);
                if (wrote_rt) {
                    route_pending = false;
                }
            }

            // If any pending remains, retry next cycle without progressing read
            if (forward_pending || route_pending) {
                continue;
            }

            // If forward aggregation is full, pack and enqueue a pending send
            if (size_forward == 8 && !forward_pending) {
                stream512u_t pkt_forward_local;
                for (int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
                    pkt_forward_local.data.range(base_bit, base_bit - 23) = dst_forward[i];
                    pkt_forward_local.data.range(base_bit - 24, base_bit - 31) = delay_forward[i];
                    pkt_forward_local.data.range(base_bit - 32, base_bit - 63) = weight_bits_forward[i];
                }
                bool wrote_fw_now = SynForwardRoute.write_nb(pkt_forward_local);
                if (wrote_fw_now) {
                    // Clear buffers on success
                    for (int i = 0; i < 8; i++) {
                        #pragma HLS UNROLL
                        dst_forward[i] = (DstID_t)0;
                        delay_forward[i] = (Delay_t)0;
                        weight_bits_forward[i] = (uint32_t)0;
                    }
                    size_forward = 0;
                } else {
                    // Defer to a future cycle
                    pending_forward_pkt = pkt_forward_local;
                    forward_pending = true;
                    continue;
                }
            }

            // Process main synapse stream
            stream512u_t pkt;
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
                // Extract all 8 synapse entries in parallel
                DstID_t dst[8];
                Delay_t delay[8];
                uint32_t weight_bits[8];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=delay complete
                #pragma HLS ARRAY_PARTITION variable=weight_bits complete
                
                // Unpack all 16 synapses at once
                for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
                    dst[i] = pkt.data.range(base_bit, base_bit - 23);
                    delay[i] = pkt.data.range(base_bit - 24, base_bit - 31);
                    weight_bits[i] = pkt.data.range(base_bit - 32, base_bit - 63);
                }

                synapse_loop: for (int i = 0; i < 8; i++) {
                    //#pragma HLS UNROLL
                    // Create synapse word
                    synapse_word_t temp;
                    temp.range(63, 40) = dst[i];
                    temp.range(39, 32) = delay[i];
                    temp.range(31, 0)  = weight_bits[i];

                    // Find region index in parallel
                    int region = -1;
                    find_region: for (int r = 0; r < 8; ++r) {
                        #pragma HLS UNROLL
                        bool is_local = (dst[i] >= start[r] && dst[i] < end[r]);
                        if (is_local && region == -1) region = r;
                    }

                    // Dispatch to the right stream (blocking write is acceptable; consumer is in DATAFLOW)
                    switch (region) {
                        case 0:  SynForward.write(temp); break;
                        case 1:  SynForward1.write(temp); break;
                        case 2:  SynForward2.write(temp); break;
                        case 3:  SynForward3.write(temp); break;
                        case 4:  SynForward4.write(temp); break;
                        case 5:  SynForward5.write(temp); break;
                        case 6:  SynForward6.write(temp); break;
                        case 7:  SynForward7.write(temp); break;
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
                        // Flush any remaining aggregated forwards without spin-wait
                        if(size_forward > 0 && !forward_pending) {
                            stream512u_t pkt_forward_local;
                            for(int i = 0; i < 8; i++) {
                                #pragma HLS UNROLL
                                int base_bit = 511 - i * 64;
                                pkt_forward_local.data.range(base_bit, base_bit - 23) = dst_forward[i];
                                pkt_forward_local.data.range(base_bit - 24, base_bit - 31) = delay_forward[i];
                                pkt_forward_local.data.range(base_bit - 32, base_bit - 63) = weight_bits_forward[i];
                            }
                            bool wrote_fw_now2 = SynForwardRoute.write_nb(pkt_forward_local);
                            if (wrote_fw_now2) {
                                for (int i = 0; i < 8; i++) {
                                    #pragma HLS UNROLL
                                    dst_forward[i] = (DstID_t)0;
                                    delay_forward[i] = (Delay_t)0;
                                    weight_bits_forward[i] = (uint32_t)0;
                                }
                                size_forward = 0;
                            } else {
                                pending_forward_pkt = pkt_forward_local;
                                forward_pending = true;
                            }
                        }
                    }

                    if(coreDone == AmountOfCores) {
                        prev_done = true;
                        if(dst[0] != NeuronStart) {
                            // Enqueue route write once; retry next cycles
                            pending_route_pkt = pkt;
                            route_pending = true;
                        }
                    } else {
                        coreDone++;
                        // Forward to next core once; retry next cycles
                        pending_route_pkt = pkt;
                        route_pending = true;
                    }
                } else {
                    // Aggregate non-local words to forward; fill until 8, flushing handled at loop top
                    for(int i = 0; i < 8; i++) {
                        if (size_forward < 8) {
                            dst_forward[size_forward] = dst[i];
                            delay_forward[size_forward] = delay[i];
                            weight_bits_forward[size_forward] = weight_bits[i];
                            size_forward++;
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
        
        SynForward.write(temp_sync);
        SynForward1.write(temp_sync);
        SynForward2.write(temp_sync);
        SynForward3.write(temp_sync);
        SynForward4.write(temp_sync);
        SynForward5.write(temp_sync);
        SynForward6.write(temp_sync);
        SynForward7.write(temp_sync);
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
    hls::stream<stream512u_t> &syn_route_in,
    hls::stream<stream512u_t> &syn_forward_rt,
    hls::stream<stream512u_t> &synapse_stream,
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

    hls::stream<ap_uint<256>> spike_out;
    hls::stream<ap_uint<256>> spike_out1;
    hls::stream<ap_uint<256>> spike_out2;
    hls::stream<ap_uint<256>> spike_out3;
    hls::stream<ap_uint<256>> spike_out4;
    hls::stream<ap_uint<256>> spike_out5;
    hls::stream<ap_uint<256>> spike_out6;
    hls::stream<ap_uint<256>> spike_out7;

#pragma HLS STREAM variable=spike_stream    depth=256
#pragma HLS STREAM variable=spike_stream1    depth=256
#pragma HLS STREAM variable=spike_stream2    depth=256
#pragma HLS STREAM variable=spike_stream3    depth=256
#pragma HLS STREAM variable=spike_stream4    depth=256
#pragma HLS STREAM variable=spike_stream5    depth=256
#pragma HLS STREAM variable=spike_stream6    depth=256
#pragma HLS STREAM variable=spike_stream7    depth=256

#pragma HLS STREAM variable=spike_out    depth=32
#pragma HLS STREAM variable=spike_out1    depth=32
#pragma HLS STREAM variable=spike_out2    depth=32
#pragma HLS STREAM variable=spike_out3    depth=32
#pragma HLS STREAM variable=spike_out4    depth=32
#pragma HLS STREAM variable=spike_out5    depth=32
#pragma HLS STREAM variable=spike_out6    depth=32
#pragma HLS STREAM variable=spike_out7    depth=32

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
        syn_forward_rt
    );

    SE0: SomaEngine(threshold, membrane_potential, 
        NeuronStart, ((NeuronTotal+7)/8), SimulationTime, spike_stream, spike_out);

    SE1: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*1), ((NeuronTotal+7)/8), SimulationTime, spike_stream1, 
        spike_out1
    );
    SE2: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*2), ((NeuronTotal+7)/8), SimulationTime, spike_stream2, 
        spike_out2
    );
    SE3: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*3), ((NeuronTotal+7)/8), SimulationTime, spike_stream3, 
        spike_out3
    );
    SE4: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*4), ((NeuronTotal+7)/8), SimulationTime, spike_stream4, 
        spike_out4
    );
    SE5: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*5), ((NeuronTotal+7)/8), SimulationTime, spike_stream5, 
        spike_out5
    );
    SE6: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*6), ((NeuronTotal+7)/8), SimulationTime, spike_stream6, 
        spike_out6
    );
    SE7: SomaEngine(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*7), ((NeuronTotal+7)/8), SimulationTime, spike_stream7, 
        spike_out7
    );

    accumulate_spike(SimulationTime, spike_out, spike_out1, spike_out2, spike_out3, spike_out4, spike_out5, spike_out6, spike_out7, spike_out_axon);
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
