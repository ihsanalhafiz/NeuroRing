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
#define BUF(core,delay) buf_flat[(core)*DELAY + (delay)]
#define BUF_IDX(core, ofs)   ((core)*DELAY + (ofs))   // ofs == head[core] or (head+delay)
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

struct lanes8_t {
    synapse_word_t data[NLANE];
};

//#pragma HLS aggregate variable=lanes8_t compact=bit

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
    int                     NeuronStart,
    int                     NeuronTotal,
    int                     SimulationTime,
    hls::stream<synapse_word_t> &SynForward,
    hls::stream<ap_uint<64>>  &SpikeOut)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    //#pragma HLS ARRAY_PARTITION variable=buf_flat complete
    //#pragma HLS ARRAY_PARTITION variable=head complete

    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    ap_uint<64> spike_status;
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

    // receive the initial packets for umempot
    for (int i = 0; i < NCORE; i++) {
        if (i < NeuronTotal) {
            synapse_word_t pkt;
            bool read_status = false;
            while(!read_status) {
                read_status = SynForward.read_nb(pkt);
            }
            float_to_uint32 weight_conv;
            weight_conv.u = pkt.range(31, 0);
            DstID_t dst = pkt.range(63, 40);
            U_membPot[dst.to_int()-NeuronStart] = weight_conv.f;
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

        for(int i = 0; i < NCORE; i++) {
            if (i < NeuronTotal) {
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
        }
        bool write_status = false;
        while(!write_status) {
            write_status = SpikeOut.write_nb(spike_status);
        }
        
        bool done = false;
        while (!done) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SynForward
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                ap_uint<6> h2 = (head[dst.to_int()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_int()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    done = true;
                } else {
                    buf_flat[BUF_IDX((dst.to_int()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }

        for(int i = 0; i < NCORE; i++) {
            if (i < NeuronTotal) {
                ap_uint<6> h3 = (head[i]) & 0x3F;
                I_PreSynCurr[i] = I_PreSynCurr[i] * I_decay + buf_flat[BUF_IDX(i,h3)];
                buf_flat[BUF_IDX(i,h3)] = 0;
                head[i] = (h3 + 1) & 0x3F;
            }
        }
    }
}

void accumulate_spike(
    int SimulationTime,
    hls::stream<ap_uint<64>> &spike_out,
    hls::stream<ap_uint<64>> &spike_out1,
    hls::stream<ap_uint<64>> &spike_out2,
    hls::stream<ap_uint<64>> &spike_out3,
    hls::stream<ap_uint<64>> &spike_out4,
    hls::stream<ap_uint<64>> &spike_out5,
    hls::stream<ap_uint<64>> &spike_out6,
    hls::stream<ap_uint<64>> &spike_out7,
    hls::stream<ap_uint<64>> &spike_out8,
    hls::stream<ap_uint<64>> &spike_out9,
    hls::stream<ap_uint<64>> &spike_out10,
    hls::stream<ap_uint<64>> &spike_out11,
    hls::stream<ap_uint<64>> &spike_out12,
    hls::stream<ap_uint<64>> &spike_out13,
    hls::stream<ap_uint<64>> &spike_out14,
    hls::stream<ap_uint<64>> &spike_out15,
    hls::stream<ap_uint<64>> &spike_out16,
    hls::stream<ap_uint<64>> &spike_out17,
    hls::stream<ap_uint<64>> &spike_out18,
    hls::stream<ap_uint<64>> &spike_out19,
    hls::stream<ap_uint<64>> &spike_out20,
    hls::stream<ap_uint<64>> &spike_out21,
    hls::stream<ap_uint<64>> &spike_out22,
    hls::stream<ap_uint<64>> &spike_out23,
    hls::stream<ap_uint<64>> &spike_out24,
    hls::stream<ap_uint<64>> &spike_out25,
    hls::stream<ap_uint<64>> &spike_out26,
    hls::stream<ap_uint<64>> &spike_out27,
    hls::stream<ap_uint<64>> &spike_out28,
    hls::stream<ap_uint<64>> &spike_out29,
    hls::stream<ap_uint<64>> &spike_out30,
    hls::stream<ap_uint<64>> &spike_out31,
    hls::stream<stream2048u_t> &spike_stream
)
{
    stream2048u_t pkt_spike;
    for (int t = 0; t < SimulationTime; t++) {
        pkt_spike.data = 0;
        ap_uint<64> spike_status;
        bool read_status = false;
        while(!read_status) {
            read_status = spike_out.read_nb(spike_status);
        }
        pkt_spike.data.range(63, 0) = spike_status;
        bool read_status1 = false;
        while(!read_status1) {
            read_status1 = spike_out1.read_nb(spike_status);
        }
        pkt_spike.data.range(127, 64) = spike_status;
        bool read_status2 = false;
        while(!read_status2) {
            read_status2 = spike_out2.read_nb(spike_status);
        }
        pkt_spike.data.range(191, 128) = spike_status;
        bool read_status3 = false;
        while(!read_status3) {
            read_status3 = spike_out3.read_nb(spike_status);
        }
        pkt_spike.data.range(255, 192) = spike_status;
        bool read_status4 = false;
        while(!read_status4) {
            read_status4 = spike_out4.read_nb(spike_status);
        }
        pkt_spike.data.range(319, 256) = spike_status;
        bool read_status5 = false;
        while(!read_status5) {
            read_status5 = spike_out5.read_nb(spike_status);
        }
        pkt_spike.data.range(383, 320) = spike_status;
        bool read_status6 = false;
        while(!read_status6) {
            read_status6 = spike_out6.read_nb(spike_status);
        }
        pkt_spike.data.range(447, 384) = spike_status;
        bool read_status7 = false;
        while(!read_status7) {
            read_status7 = spike_out7.read_nb(spike_status);
        }
        pkt_spike.data.range(511, 448) = spike_status;  
        bool read_status8 = false;
        while(!read_status8) {
            read_status8 = spike_out8.read_nb(spike_status);
        }
        pkt_spike.data.range(575, 512) = spike_status;
        bool read_status9 = false;
        while(!read_status9) {
            read_status9 = spike_out9.read_nb(spike_status);
        }
        pkt_spike.data.range(639, 576) = spike_status;
        bool read_status10 = false;
        while(!read_status10) {
            read_status10 = spike_out10.read_nb(spike_status);
        }
        pkt_spike.data.range(703, 640) = spike_status;
        bool read_status11 = false;
        while(!read_status11) {
            read_status11 = spike_out11.read_nb(spike_status);
        }
        pkt_spike.data.range(767, 704) = spike_status;
        bool read_status12 = false;
        while(!read_status12) {
            read_status12 = spike_out12.read_nb(spike_status);
        }
        pkt_spike.data.range(831, 768) = spike_status;
        bool read_status13 = false;
        while(!read_status13) {
            read_status13 = spike_out13.read_nb(spike_status);
        }
        pkt_spike.data.range(895, 832) = spike_status;
        bool read_status14 = false;
        while(!read_status14) {
            read_status14 = spike_out14.read_nb(spike_status);
        }
        pkt_spike.data.range(959, 896) = spike_status;
        bool read_status15 = false;
        while(!read_status15) {
            read_status15 = spike_out15.read_nb(spike_status);
        }
        pkt_spike.data.range(1023, 960) = spike_status;
        bool read_status16 = false;
        while(!read_status16) {
            read_status16 = spike_out16.read_nb(spike_status);
        }
        pkt_spike.data.range(1087, 1024) = spike_status;
        bool read_status17 = false;
        while(!read_status17) {
            read_status17 = spike_out17.read_nb(spike_status);
        }
        pkt_spike.data.range(1151, 1088) = spike_status;
        bool read_status18 = false;
        while(!read_status18) {
            read_status18 = spike_out18.read_nb(spike_status);
        }
        pkt_spike.data.range(1215, 1152) = spike_status;    
        bool read_status19 = false;
        while(!read_status19) {
            read_status19 = spike_out19.read_nb(spike_status);
        }
        pkt_spike.data.range(1279, 1216) = spike_status;
        bool read_status20 = false;
        while(!read_status20) {
            read_status20 = spike_out20.read_nb(spike_status);
        }
        pkt_spike.data.range(1343, 1280) = spike_status;
        bool read_status21 = false;
        while(!read_status21) {
            read_status21 = spike_out21.read_nb(spike_status);
        }
        pkt_spike.data.range(1407, 1344) = spike_status;
        bool read_status22 = false;
        while(!read_status22) {
            read_status22 = spike_out22.read_nb(spike_status);
        }
        pkt_spike.data.range(1471, 1408) = spike_status;
        bool read_status23 = false;
        while(!read_status23) {
            read_status23 = spike_out23.read_nb(spike_status);
        }
        pkt_spike.data.range(1535, 1472) = spike_status;
        bool read_status24 = false;
        while(!read_status24) {
            read_status24 = spike_out24.read_nb(spike_status);
        }
        pkt_spike.data.range(1599, 1536) = spike_status;
        bool read_status25 = false;
        while(!read_status25) {
            read_status25 = spike_out25.read_nb(spike_status);
        }
        pkt_spike.data.range(1663, 1600) = spike_status;
        bool read_status26 = false;
        while(!read_status26) {
            read_status26 = spike_out26.read_nb(spike_status);
        }
        pkt_spike.data.range(1727, 1664) = spike_status;
        bool read_status27 = false;
        while(!read_status27) {
            read_status27 = spike_out27.read_nb(spike_status);
        }
        pkt_spike.data.range(1791, 1728) = spike_status;
        bool read_status28 = false;
        while(!read_status28) {
            read_status28 = spike_out28.read_nb(spike_status);
        }
        pkt_spike.data.range(1855, 1792) = spike_status;
        bool read_status29 = false;
        while(!read_status29) {
            read_status29 = spike_out29.read_nb(spike_status);
        }
        pkt_spike.data.range(1919, 1856) = spike_status;
        bool read_status30 = false;
        while(!read_status30) {
            read_status30 = spike_out30.read_nb(spike_status);
        }
        pkt_spike.data.range(1983, 1920) = spike_status;
        bool read_status31 = false;
        while(!read_status31) {
            read_status31 = spike_out31.read_nb(spike_status);
        }
        pkt_spike.data.range(2047, 1984) = spike_status;

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
    hls::stream<stream2048u_t> &SynapseStream,
    hls::stream<stream2048u_t> &SynapseStreamRoute,
    int                     NeuronStart,
    int                     NeuronTotal,
    int                     SimulationTime,
    int                     AmountOfCores,
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
    hls::stream<synapse_word_t> &SynForward16,
    hls::stream<synapse_word_t> &SynForward17,
    hls::stream<synapse_word_t> &SynForward18,
    hls::stream<synapse_word_t> &SynForward19,
    hls::stream<synapse_word_t> &SynForward20,
    hls::stream<synapse_word_t> &SynForward21,
    hls::stream<synapse_word_t> &SynForward22,
    hls::stream<synapse_word_t> &SynForward23,
    hls::stream<synapse_word_t> &SynForward24,
    hls::stream<synapse_word_t> &SynForward25,
    hls::stream<synapse_word_t> &SynForward26,
    hls::stream<synapse_word_t> &SynForward27,
    hls::stream<synapse_word_t> &SynForward28,
    hls::stream<synapse_word_t> &SynForward29,
    hls::stream<synapse_word_t> &SynForward30,
    hls::stream<synapse_word_t> &SynForward31,
    hls::stream<stream2048u_t> &SynForwardRoute)
{
    // Helper function to write packet to stream
    auto write_synapse_nb = [](hls::stream<synapse_word_t>& stream, const synapse_word_t& packet, bool &is_local) {
        bool write_status = false;
        while (!write_status && is_local) {
            write_status = stream.write_nb(packet);
        }
    };

    for (int i = 0; i < NEURON_NUM; i+=32) {
        if (i < NeuronTotal) {
            stream2048u_t pkt;
            bool read_status = false;
            while(!read_status) {
                read_status = SynapseStream.read_nb(pkt);
            }
            DstID_t dst[32];
            Delay_t delay[32];
            uint32_t weight_bits[32];
            #pragma HLS ARRAY_PARTITION variable=dst complete
            #pragma HLS ARRAY_PARTITION variable=delay complete
            #pragma HLS ARRAY_PARTITION variable=weight_bits complete
            for (int j = 0; j < 32; j++) {
                int base_bit = 2047 - j * 64;
                dst[j] = pkt.data.range(base_bit, base_bit - 23);
                delay[j] = pkt.data.range(base_bit - 24, base_bit - 31);
                weight_bits[j] = pkt.data.range(base_bit - 32, base_bit - 63);
            }
            for (int j = 0; j < 32; j++) {
                synapse_word_t temp;
                temp.range(63, 40) = dst[j];
                temp.range(39, 32) = delay[j];
                temp.range(31, 0) = weight_bits[j];
                bool is_local0 = (dst[j] >= 0+NeuronStart && dst[j] < 64+NeuronStart);
                bool is_local1 = (dst[j] >= 64+NeuronStart && dst[j] < 128+NeuronStart);
                bool is_local2 = (dst[j] >= 128+NeuronStart && dst[j] < 192+NeuronStart);
                bool is_local3 = (dst[j] >= 192+NeuronStart && dst[j] < 256+NeuronStart);
                bool is_local4 = (dst[j] >= 256+NeuronStart && dst[j] < 320+NeuronStart);
                bool is_local5 = (dst[j] >= 320+NeuronStart && dst[j] < 384+NeuronStart);
                bool is_local6 = (dst[j] >= 384+NeuronStart && dst[j] < 448+NeuronStart);
                bool is_local7 = (dst[j] >= 448+NeuronStart && dst[j] < 512+NeuronStart);
                bool is_local8 = (dst[j] >= 512+NeuronStart && dst[j] < 576+NeuronStart);
                bool is_local9 = (dst[j] >= 576+NeuronStart && dst[j] < 640+NeuronStart);
                bool is_local10 = (dst[j] >= 640+NeuronStart && dst[j] < 704+NeuronStart);
                bool is_local11 = (dst[j] >= 704+NeuronStart && dst[j] < 768+NeuronStart);
                bool is_local12 = (dst[j] >= 768+NeuronStart && dst[j] < 832+NeuronStart);
                bool is_local13 = (dst[j] >= 832+NeuronStart && dst[j] < 896+NeuronStart);
                bool is_local14 = (dst[j] >= 896+NeuronStart && dst[j] < 960+NeuronStart);
                bool is_local15 = (dst[j] >= 960+NeuronStart && dst[j] < 1024+NeuronStart);
                bool is_local16 = (dst[j] >= 1024+NeuronStart && dst[j] < 1088+NeuronStart);
                bool is_local17 = (dst[j] >= 1088+NeuronStart && dst[j] < 1152+NeuronStart);
                bool is_local18 = (dst[j] >= 1152+NeuronStart && dst[j] < 1216+NeuronStart);
                bool is_local19 = (dst[j] >= 1216+NeuronStart && dst[j] < 1280+NeuronStart);
                bool is_local20 = (dst[j] >= 1280+NeuronStart && dst[j] < 1344+NeuronStart);
                bool is_local21 = (dst[j] >= 1344+NeuronStart && dst[j] < 1408+NeuronStart);
                bool is_local22 = (dst[j] >= 1408+NeuronStart && dst[j] < 1472+NeuronStart);
                bool is_local23 = (dst[j] >= 1472+NeuronStart && dst[j] < 1536+NeuronStart);
                bool is_local24 = (dst[j] >= 1536+NeuronStart && dst[j] < 1600+NeuronStart);
                bool is_local25 = (dst[j] >= 1600+NeuronStart && dst[j] < 1664+NeuronStart);
                bool is_local26 = (dst[j] >= 1664+NeuronStart && dst[j] < 1728+NeuronStart);
                bool is_local27 = (dst[j] >= 1728+NeuronStart && dst[j] < 1792+NeuronStart);
                bool is_local28 = (dst[j] >= 1792+NeuronStart && dst[j] < 1856+NeuronStart);
                bool is_local29 = (dst[j] >= 1856+NeuronStart && dst[j] < 1920+NeuronStart);
                bool is_local30 = (dst[j] >= 1920+NeuronStart && dst[j] < 1984+NeuronStart);
                bool is_local31 = (dst[j] >= 1984+NeuronStart && dst[j] < 2048+NeuronStart);
                if(is_local0){SynForward.write(temp);}
                else if(is_local1){SynForward1.write(temp);}
                else if(is_local2){SynForward2.write(temp);}
                else if(is_local3){SynForward3.write(temp);}
                else if(is_local4){SynForward4.write(temp);}
                else if(is_local5){SynForward5.write(temp);}
                else if(is_local6){SynForward6.write(temp);}
                else if(is_local7){SynForward7.write(temp);}
                else if(is_local8){SynForward8.write(temp);}
                else if(is_local9){SynForward9.write(temp);}
                else if(is_local10){SynForward10.write(temp);}
                else if(is_local11){SynForward11.write(temp);}
                else if(is_local12){SynForward12.write(temp);}
                else if(is_local13){SynForward13.write(temp);}
                else if(is_local14){SynForward14.write(temp);}
                else if(is_local15){SynForward15.write(temp);}
                else if(is_local16){SynForward16.write(temp);}
                else if(is_local17){SynForward17.write(temp);}
                else if(is_local18){SynForward18.write(temp);}
                else if(is_local19){SynForward19.write(temp);}
                else if(is_local20){SynForward20.write(temp);}
                else if(is_local21){SynForward21.write(temp);}
                else if(is_local22){SynForward22.write(temp);}
                else if(is_local23){SynForward23.write(temp);}
                else if(is_local24){SynForward24.write(temp);}
                else if(is_local25){SynForward25.write(temp);}
                else if(is_local26){SynForward26.write(temp);}
                else if(is_local27){SynForward27.write(temp);}
                else if(is_local28){SynForward28.write(temp);}
                else if(is_local29){SynForward29.write(temp);}
                else if(is_local30){SynForward30.write(temp);}
                else if(is_local31){SynForward31.write(temp);}            
            }
        }
    }
    
    router_loop: for (int t = 0; t < SimulationTime; t++) {
        bool axon_done = false;
        bool prev_done = false;
        uint32_t coreDone = 0;
        
        while (!(axon_done && prev_done)) {
        #pragma HLS PIPELINE II=1 rewind
            
            // Process main synapse stream
            stream2048u_t pkt;
            bool have_pkt = SynapseStream.read_nb(pkt);
            if (have_pkt) {
                // Extract all 8 synapse entries in parallel
                DstID_t dst[32];
                Delay_t delay[32];
                uint32_t weight_bits[32];
                bool status_neuron[32];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=delay complete
                #pragma HLS ARRAY_PARTITION variable=weight_bits complete
                #pragma HLS ARRAY_PARTITION variable=status_neuron complete
                
                // Unpack all 8 synapses at once
                for (int i = 0; i < 32; i++) {
                #pragma HLS UNROLL
                    int base_bit = 2047 - i * 64;
                    dst[i] = pkt.data.range(base_bit, base_bit - 23);
                    delay[i] = pkt.data.range(base_bit - 24, base_bit - 31);
                    weight_bits[i] = pkt.data.range(base_bit - 32, base_bit - 63);
                }
                
                // Check if this is an axon done signal
                if (delay[0] == 0xFE) {
                    axon_done = true;
                    // Non-blocking write with retry
                    SynForwardRoute.write(pkt);
                } else {
                    // Process all 8 synapses efficiently
                    synapse_loop: for (int i = 0; i < 32; i++) {
                        // Create synapse word
                        synapse_word_t temp;
                        temp.range(63, 40) = dst[i];
                        temp.range(39, 32) = delay[i];
                        temp.range(31, 0) = weight_bits[i];
                        
                        // Route based on destination
                        bool is_local0 = (dst[i] >= 0+NeuronStart && dst[i] < 64+NeuronStart);
                        bool is_local1 = (dst[i] >= 64+NeuronStart && dst[i] < 128+NeuronStart);
                        bool is_local2 = (dst[i] >= 128+NeuronStart && dst[i] < 192+NeuronStart);
                        bool is_local3 = (dst[i] >= 192+NeuronStart && dst[i] < 256+NeuronStart);
                        bool is_local4 = (dst[i] >= 256+NeuronStart && dst[i] < 320+NeuronStart);
                        bool is_local5 = (dst[i] >= 320+NeuronStart && dst[i] < 384+NeuronStart);
                        bool is_local6 = (dst[i] >= 384+NeuronStart && dst[i] < 448+NeuronStart);
                        bool is_local7 = (dst[i] >= 448+NeuronStart && dst[i] < 512+NeuronStart);
                        bool is_local8 = (dst[i] >= 512+NeuronStart && dst[i] < 576+NeuronStart);
                        bool is_local9 = (dst[i] >= 576+NeuronStart && dst[i] < 640+NeuronStart);
                        bool is_local10 = (dst[i] >= 640+NeuronStart && dst[i] < 704+NeuronStart);
                        bool is_local11 = (dst[i] >= 704+NeuronStart && dst[i] < 768+NeuronStart);
                        bool is_local12 = (dst[i] >= 768+NeuronStart && dst[i] < 832+NeuronStart);
                        bool is_local13 = (dst[i] >= 832+NeuronStart && dst[i] < 896+NeuronStart);
                        bool is_local14 = (dst[i] >= 896+NeuronStart && dst[i] < 960+NeuronStart);
                        bool is_local15 = (dst[i] >= 960+NeuronStart && dst[i] < 1024+NeuronStart);
                        bool is_local16 = (dst[i] >= 1024+NeuronStart && dst[i] < 1088+NeuronStart);
                        bool is_local17 = (dst[i] >= 1088+NeuronStart && dst[i] < 1152+NeuronStart);
                        bool is_local18 = (dst[i] >= 1152+NeuronStart && dst[i] < 1216+NeuronStart);
                        bool is_local19 = (dst[i] >= 1216+NeuronStart && dst[i] < 1280+NeuronStart);
                        bool is_local20 = (dst[i] >= 1280+NeuronStart && dst[i] < 1344+NeuronStart);
                        bool is_local21 = (dst[i] >= 1344+NeuronStart && dst[i] < 1408+NeuronStart);
                        bool is_local22 = (dst[i] >= 1408+NeuronStart && dst[i] < 1472+NeuronStart);
                        bool is_local23 = (dst[i] >= 1472+NeuronStart && dst[i] < 1536+NeuronStart);
                        bool is_local24 = (dst[i] >= 1536+NeuronStart && dst[i] < 1600+NeuronStart);
                        bool is_local25 = (dst[i] >= 1600+NeuronStart && dst[i] < 1664+NeuronStart);
                        bool is_local26 = (dst[i] >= 1664+NeuronStart && dst[i] < 1728+NeuronStart);
                        bool is_local27 = (dst[i] >= 1728+NeuronStart && dst[i] < 1792+NeuronStart);
                        bool is_local28 = (dst[i] >= 1792+NeuronStart && dst[i] < 1856+NeuronStart);
                        bool is_local29 = (dst[i] >= 1856+NeuronStart && dst[i] < 1920+NeuronStart);
                        bool is_local30 = (dst[i] >= 1920+NeuronStart && dst[i] < 1984+NeuronStart);
                        bool is_local31 = (dst[i] >= 1984+NeuronStart && dst[i] < 2048+NeuronStart);
                        bool del_data = false;
                        if(is_local0){SynForward.write(temp); del_data = true;}
                        else if(is_local1){SynForward1.write(temp); del_data = true;}
                        else if(is_local2){SynForward2.write(temp); del_data = true;}
                        else if(is_local3){SynForward3.write(temp); del_data = true;}
                        else if(is_local4){SynForward4.write(temp); del_data = true;}
                        else if(is_local5){SynForward5.write(temp); del_data = true;}
                        else if(is_local6){SynForward6.write(temp); del_data = true;}
                        else if(is_local7){SynForward7.write(temp); del_data = true;}
                        else if(is_local8){SynForward8.write(temp); del_data = true;}
                        else if(is_local9){SynForward9.write(temp); del_data = true;}
                        else if(is_local10){SynForward10.write(temp); del_data = true;}
                        else if(is_local11){SynForward11.write(temp); del_data = true;}
                        else if(is_local12){SynForward12.write(temp); del_data = true;}
                        else if(is_local13){SynForward13.write(temp); del_data = true;}
                        else if(is_local14){SynForward14.write(temp); del_data = true;}
                        else if(is_local15){SynForward15.write(temp); del_data = true;}
                        else if(is_local16){SynForward16.write(temp); del_data = true;}
                        else if(is_local17){SynForward17.write(temp); del_data = true;}
                        else if(is_local18){SynForward18.write(temp); del_data = true;}
                        else if(is_local19){SynForward19.write(temp); del_data = true;}
                        else if(is_local20){SynForward20.write(temp); del_data = true;}
                        else if(is_local21){SynForward21.write(temp); del_data = true;}
                        else if(is_local22){SynForward22.write(temp); del_data = true;}
                        else if(is_local23){SynForward23.write(temp); del_data = true;}
                        else if(is_local24){SynForward24.write(temp); del_data = true;}
                        else if(is_local25){SynForward25.write(temp); del_data = true;}
                        else if(is_local26){SynForward26.write(temp); del_data = true;}
                        else if(is_local27){SynForward27.write(temp); del_data = true;}
                        else if(is_local28){SynForward28.write(temp); del_data = true;}
                        else if(is_local29){SynForward29.write(temp); del_data = true;}
                        else if(is_local30){SynForward30.write(temp); del_data = true;}
                        else if(is_local31){SynForward31.write(temp); del_data = true;}

                        if(del_data) {
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        }
                    }
                    bool any_non_zero = false;
                    for(int i = 0; i < 32; i++) {
                        any_non_zero = any_non_zero || (dst[i] != 0);
                    }
                    if(any_non_zero) {
                        // create stream512u_t packet
                        stream2048u_t temp_pkt;
                        for(int i = 0; i < 32; i++) {
                            #pragma HLS UNROLL
                            int base_bit = 2047 - i * 64;
                            temp_pkt.data.range(base_bit, base_bit - 23) = dst[i];
                            temp_pkt.data.range(base_bit - 24, base_bit - 31) = delay[i];
                            temp_pkt.data.range(base_bit - 32, base_bit - 63) = weight_bits[i];
                        }
                        SynForwardRoute.write(temp_pkt);
                    }
                }
            }
            
            // Process routed stream from previous router
            stream2048u_t temp_rt;
            bool have_rt = SynapseStreamRoute.read_nb(temp_rt);
            if (have_rt) {
                DstID_t dst[32];
                Delay_t delay[32];
                uint32_t weight_bits[32];
                bool status_neuron[32];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=delay complete
                #pragma HLS ARRAY_PARTITION variable=weight_bits complete
                #pragma HLS ARRAY_PARTITION variable=status_neuron complete

                for(int i = 0; i < 32; i++) {
                    #pragma HLS UNROLL
                    int base_bit = 2047 - i * 64;
                    dst[i] = temp_rt.data.range(base_bit, base_bit - 23);
                    delay[i] = temp_rt.data.range(base_bit - 24, base_bit - 31);
                    weight_bits[i] = temp_rt.data.range(base_bit - 32, base_bit - 63);
                }

                if (delay[0] == 0xFE) {
                    // Handle synchronization
                    if(coreDone == AmountOfCores - 1) {
                        prev_done = true;
                    } else {
                        coreDone++;
                    }
                    // Forward if not for this core
                    if (dst[0] != ap_uint<24>(NeuronStart)) {
                        SynForwardRoute.write(temp_rt);
                    }
                } else {
                    // Route based on destination
                    synapse_loop1: for (int i = 0; i < 32; i++) {
                        // Create synapse word
                        synapse_word_t temp;
                        temp.range(63, 40) = dst[i];
                        temp.range(39, 32) = delay[i];
                        temp.range(31, 0) = weight_bits[i];
                        
                        // Route based on destination
                        bool is_local0 = (dst[i] >= 0+NeuronStart && dst[i] < 64+NeuronStart);
                        bool is_local1 = (dst[i] >= 64+NeuronStart && dst[i] < 128+NeuronStart);
                        bool is_local2 = (dst[i] >= 128+NeuronStart && dst[i] < 192+NeuronStart);
                        bool is_local3 = (dst[i] >= 192+NeuronStart && dst[i] < 256+NeuronStart);
                        bool is_local4 = (dst[i] >= 256+NeuronStart && dst[i] < 320+NeuronStart);
                        bool is_local5 = (dst[i] >= 320+NeuronStart && dst[i] < 384+NeuronStart);
                        bool is_local6 = (dst[i] >= 384+NeuronStart && dst[i] < 448+NeuronStart);
                        bool is_local7 = (dst[i] >= 448+NeuronStart && dst[i] < 512+NeuronStart);
                        bool is_local8 = (dst[i] >= 512+NeuronStart && dst[i] < 576+NeuronStart);
                        bool is_local9 = (dst[i] >= 576+NeuronStart && dst[i] < 640+NeuronStart);
                        bool is_local10 = (dst[i] >= 640+NeuronStart && dst[i] < 704+NeuronStart);
                        bool is_local11 = (dst[i] >= 704+NeuronStart && dst[i] < 768+NeuronStart);
                        bool is_local12 = (dst[i] >= 768+NeuronStart && dst[i] < 832+NeuronStart);
                        bool is_local13 = (dst[i] >= 832+NeuronStart && dst[i] < 896+NeuronStart);
                        bool is_local14 = (dst[i] >= 896+NeuronStart && dst[i] < 960+NeuronStart);
                        bool is_local15 = (dst[i] >= 960+NeuronStart && dst[i] < 1024+NeuronStart);
                        bool is_local16 = (dst[i] >= 1024+NeuronStart && dst[i] < 1088+NeuronStart);
                        bool is_local17 = (dst[i] >= 1088+NeuronStart && dst[i] < 1152+NeuronStart);
                        bool is_local18 = (dst[i] >= 1152+NeuronStart && dst[i] < 1216+NeuronStart);
                        bool is_local19 = (dst[i] >= 1216+NeuronStart && dst[i] < 1280+NeuronStart);
                        bool is_local20 = (dst[i] >= 1280+NeuronStart && dst[i] < 1344+NeuronStart);
                        bool is_local21 = (dst[i] >= 1344+NeuronStart && dst[i] < 1408+NeuronStart);
                        bool is_local22 = (dst[i] >= 1408+NeuronStart && dst[i] < 1472+NeuronStart);
                        bool is_local23 = (dst[i] >= 1472+NeuronStart && dst[i] < 1536+NeuronStart);
                        bool is_local24 = (dst[i] >= 1536+NeuronStart && dst[i] < 1600+NeuronStart);
                        bool is_local25 = (dst[i] >= 1600+NeuronStart && dst[i] < 1664+NeuronStart);
                        bool is_local26 = (dst[i] >= 1664+NeuronStart && dst[i] < 1728+NeuronStart);
                        bool is_local27 = (dst[i] >= 1728+NeuronStart && dst[i] < 1792+NeuronStart);
                        bool is_local28 = (dst[i] >= 1792+NeuronStart && dst[i] < 1856+NeuronStart);
                        bool is_local29 = (dst[i] >= 1856+NeuronStart && dst[i] < 1920+NeuronStart);
                        bool is_local30 = (dst[i] >= 1920+NeuronStart && dst[i] < 1984+NeuronStart);
                        bool is_local31 = (dst[i] >= 1984+NeuronStart && dst[i] < 2048+NeuronStart);
                        bool del_data = false;
                        if(is_local0){SynForward.write(temp); del_data = true;}
                        else if(is_local1){SynForward1.write(temp); del_data = true;}
                        else if(is_local2){SynForward2.write(temp); del_data = true;}
                        else if(is_local3){SynForward3.write(temp); del_data = true;}
                        else if(is_local4){SynForward4.write(temp); del_data = true;}
                        else if(is_local5){SynForward5.write(temp); del_data = true;}
                        else if(is_local6){SynForward6.write(temp); del_data = true;}
                        else if(is_local7){SynForward7.write(temp); del_data = true;}
                        else if(is_local8){SynForward8.write(temp); del_data = true;}
                        else if(is_local9){SynForward9.write(temp); del_data = true;}
                        else if(is_local10){SynForward10.write(temp); del_data = true;}
                        else if(is_local11){SynForward11.write(temp); del_data = true;}
                        else if(is_local12){SynForward12.write(temp); del_data = true;}
                        else if(is_local13){SynForward13.write(temp); del_data = true;}
                        else if(is_local14){SynForward14.write(temp); del_data = true;}
                        else if(is_local15){SynForward15.write(temp); del_data = true;}
                        else if(is_local16){SynForward16.write(temp); del_data = true;}
                        else if(is_local17){SynForward17.write(temp); del_data = true;}
                        else if(is_local18){SynForward18.write(temp); del_data = true;}
                        else if(is_local19){SynForward19.write(temp); del_data = true;}
                        else if(is_local20){SynForward20.write(temp); del_data = true;}
                        else if(is_local21){SynForward21.write(temp); del_data = true;}
                        else if(is_local22){SynForward22.write(temp); del_data = true;}
                        else if(is_local23){SynForward23.write(temp); del_data = true;}
                        else if(is_local24){SynForward24.write(temp); del_data = true;}
                        else if(is_local25){SynForward25.write(temp); del_data = true;}
                        else if(is_local26){SynForward26.write(temp); del_data = true;}
                        else if(is_local27){SynForward27.write(temp); del_data = true;}
                        else if(is_local28){SynForward28.write(temp); del_data = true;}
                        else if(is_local29){SynForward29.write(temp); del_data = true;}
                        else if(is_local30){SynForward30.write(temp); del_data = true;}
                        else if(is_local31){SynForward31.write(temp); del_data = true;}

                        if(del_data) {
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        }
                    }
                    bool any_non_zero = false;
                    for(int i = 0; i < 32; i++) {
                        any_non_zero = any_non_zero || (dst[i] != 0);
                    }
                    if(any_non_zero) {                        // create stream512u_t packet
                        stream2048u_t temp_pkt;
                        for(int i = 0; i < 32; i++) {
                            #pragma HLS UNROLL
                            int base_bit = 2047 - i * 64;
                            temp_pkt.data.range(base_bit, base_bit - 23) = dst[i];
                            temp_pkt.data.range(base_bit - 24, base_bit - 31) = delay[i];
                            temp_pkt.data.range(base_bit - 32, base_bit - 63) = weight_bits[i];
                        }
                        SynForwardRoute.write(temp_pkt);
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
        SynForward8.write(temp_sync);
        SynForward9.write(temp_sync);
        SynForward10.write(temp_sync);
        SynForward11.write(temp_sync);
        SynForward12.write(temp_sync);
        SynForward13.write(temp_sync);
        SynForward14.write(temp_sync);
        SynForward15.write(temp_sync);
        SynForward16.write(temp_sync);
        SynForward17.write(temp_sync);
        SynForward18.write(temp_sync);
        SynForward19.write(temp_sync);
        SynForward20.write(temp_sync);
        SynForward21.write(temp_sync);
        SynForward22.write(temp_sync);
        SynForward23.write(temp_sync);
        SynForward24.write(temp_sync);
        SynForward25.write(temp_sync);
        SynForward26.write(temp_sync);
        SynForward27.write(temp_sync);
        SynForward28.write(temp_sync);
        SynForward29.write(temp_sync);
        SynForward30.write(temp_sync);
        SynForward31.write(temp_sync);
    }
}
//--------------------------------------------------------------------
//  Top‑level kernel ‒ integrates all sub‑kernels using DATAFLOW
//--------------------------------------------------------------------
extern "C" void NeuroRing(
    int              SimulationTime,
    uint32_t         threshold,
    uint32_t         membrane_potential,
    int              AmountOfCores,
    int              NeuronStart,
    int              NeuronTotal,
    hls::stream<stream2048u_t> &syn_route_in,
    hls::stream<stream2048u_t> &syn_forward_rt,
    hls::stream<stream2048u_t> &synapse_stream,
    hls::stream<stream2048u_t> &spike_stream
)
{
#pragma HLS INTERFACE axis port=synapse_stream bundle=AXIS_IN
#pragma HLS INTERFACE axis port=spike_stream bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=syn_route_in bundle=AXIS_IN
#pragma HLS INTERFACE axis port=syn_forward_rt bundle=AXIS_OUT

//---------------------------
//  On‑chip FIFO channels
//---------------------------
#pragma HLS DATAFLOW
    hls::stream<synapse_word_t> syn_forward;
    hls::stream<synapse_word_t> syn_forward1;
    hls::stream<synapse_word_t> syn_forward2;
    hls::stream<synapse_word_t> syn_forward3;
    hls::stream<synapse_word_t> syn_forward4;
    hls::stream<synapse_word_t> syn_forward5;
    hls::stream<synapse_word_t> syn_forward6;
    hls::stream<synapse_word_t> syn_forward7;
    hls::stream<synapse_word_t> syn_forward8;
    hls::stream<synapse_word_t> syn_forward9;
    hls::stream<synapse_word_t> syn_forward10;
    hls::stream<synapse_word_t> syn_forward11;
    hls::stream<synapse_word_t> syn_forward12;
    hls::stream<synapse_word_t> syn_forward13;
    hls::stream<synapse_word_t> syn_forward14;
    hls::stream<synapse_word_t> syn_forward15;
    hls::stream<synapse_word_t> syn_forward16;
    hls::stream<synapse_word_t> syn_forward17;
    hls::stream<synapse_word_t> syn_forward18;
    hls::stream<synapse_word_t> syn_forward19;
    hls::stream<synapse_word_t> syn_forward20;
    hls::stream<synapse_word_t> syn_forward21;
    hls::stream<synapse_word_t> syn_forward22;
    hls::stream<synapse_word_t> syn_forward23;
    hls::stream<synapse_word_t> syn_forward24;
    hls::stream<synapse_word_t> syn_forward25;
    hls::stream<synapse_word_t> syn_forward26;
    hls::stream<synapse_word_t> syn_forward27;
    hls::stream<synapse_word_t> syn_forward28;
    hls::stream<synapse_word_t> syn_forward29;
    hls::stream<synapse_word_t> syn_forward30;
    hls::stream<synapse_word_t> syn_forward31;

#pragma HLS STREAM variable=syn_forward     depth=256
#pragma HLS STREAM variable=syn_forward1    depth=256
#pragma HLS STREAM variable=syn_forward2    depth=256
#pragma HLS STREAM variable=syn_forward3    depth=256
#pragma HLS STREAM variable=syn_forward4    depth=256
#pragma HLS STREAM variable=syn_forward5    depth=256
#pragma HLS STREAM variable=syn_forward6    depth=256
#pragma HLS STREAM variable=syn_forward7    depth=256
#pragma HLS STREAM variable=syn_forward8    depth=256
#pragma HLS STREAM variable=syn_forward9    depth=256
#pragma HLS STREAM variable=syn_forward10   depth=256
#pragma HLS STREAM variable=syn_forward11   depth=256
#pragma HLS STREAM variable=syn_forward12   depth=256
#pragma HLS STREAM variable=syn_forward13   depth=256
#pragma HLS STREAM variable=syn_forward14   depth=256
#pragma HLS STREAM variable=syn_forward15   depth=256
#pragma HLS STREAM variable=syn_forward16   depth=256
#pragma HLS STREAM variable=syn_forward17   depth=256
#pragma HLS STREAM variable=syn_forward18   depth=256
#pragma HLS STREAM variable=syn_forward19   depth=256
#pragma HLS STREAM variable=syn_forward20   depth=256
#pragma HLS STREAM variable=syn_forward21   depth=256
#pragma HLS STREAM variable=syn_forward22   depth=256
#pragma HLS STREAM variable=syn_forward23   depth=256
#pragma HLS STREAM variable=syn_forward24   depth=256
#pragma HLS STREAM variable=syn_forward25   depth=256
#pragma HLS STREAM variable=syn_forward26   depth=256
#pragma HLS STREAM variable=syn_forward27   depth=256
#pragma HLS STREAM variable=syn_forward28   depth=256
#pragma HLS STREAM variable=syn_forward29   depth=256
#pragma HLS STREAM variable=syn_forward30   depth=256
#pragma HLS STREAM variable=syn_forward31   depth=256

    hls::stream<ap_uint<64>> spike_out;
    hls::stream<ap_uint<64>> spike_out1;
    hls::stream<ap_uint<64>> spike_out2;
    hls::stream<ap_uint<64>> spike_out3;
    hls::stream<ap_uint<64>> spike_out4;
    hls::stream<ap_uint<64>> spike_out5;
    hls::stream<ap_uint<64>> spike_out6;
    hls::stream<ap_uint<64>> spike_out7;
    hls::stream<ap_uint<64>> spike_out8;
    hls::stream<ap_uint<64>> spike_out9;
    hls::stream<ap_uint<64>> spike_out10;
    hls::stream<ap_uint<64>> spike_out11;
    hls::stream<ap_uint<64>> spike_out12;
    hls::stream<ap_uint<64>> spike_out13;
    hls::stream<ap_uint<64>> spike_out14;
    hls::stream<ap_uint<64>> spike_out15;
    hls::stream<ap_uint<64>> spike_out16;
    hls::stream<ap_uint<64>> spike_out17;
    hls::stream<ap_uint<64>> spike_out18;
    hls::stream<ap_uint<64>> spike_out19;
    hls::stream<ap_uint<64>> spike_out20;
    hls::stream<ap_uint<64>> spike_out21;
    hls::stream<ap_uint<64>> spike_out22;
    hls::stream<ap_uint<64>> spike_out23;
    hls::stream<ap_uint<64>> spike_out24;
    hls::stream<ap_uint<64>> spike_out25;
    hls::stream<ap_uint<64>> spike_out26;
    hls::stream<ap_uint<64>> spike_out27;
    hls::stream<ap_uint<64>> spike_out28;
    hls::stream<ap_uint<64>> spike_out29;
    hls::stream<ap_uint<64>> spike_out30;
    hls::stream<ap_uint<64>> spike_out31;

    #pragma HLS STREAM variable=spike_out    depth=16
    #pragma HLS STREAM variable=spike_out1    depth=16
    #pragma HLS STREAM variable=spike_out2    depth=16
    #pragma HLS STREAM variable=spike_out3    depth=16
    #pragma HLS STREAM variable=spike_out4    depth=16
    #pragma HLS STREAM variable=spike_out5    depth=16
    #pragma HLS STREAM variable=spike_out6    depth=16
    #pragma HLS STREAM variable=spike_out7    depth=16
    #pragma HLS STREAM variable=spike_out8    depth=16
    #pragma HLS STREAM variable=spike_out9    depth=16
    #pragma HLS STREAM variable=spike_out10   depth=16
    #pragma HLS STREAM variable=spike_out11   depth=16
    #pragma HLS STREAM variable=spike_out12   depth=16
    #pragma HLS STREAM variable=spike_out13   depth=16
    #pragma HLS STREAM variable=spike_out14   depth=16
    #pragma HLS STREAM variable=spike_out15   depth=16
    #pragma HLS STREAM variable=spike_out16   depth=16
    #pragma HLS STREAM variable=spike_out17   depth=16
    #pragma HLS STREAM variable=spike_out18   depth=16
    #pragma HLS STREAM variable=spike_out19   depth=16
    #pragma HLS STREAM variable=spike_out20   depth=16
    #pragma HLS STREAM variable=spike_out21   depth=16
    #pragma HLS STREAM variable=spike_out22   depth=16
    #pragma HLS STREAM variable=spike_out23   depth=16
    #pragma HLS STREAM variable=spike_out24   depth=16
    #pragma HLS STREAM variable=spike_out25   depth=16
    #pragma HLS STREAM variable=spike_out26   depth=16
    #pragma HLS STREAM variable=spike_out27   depth=16
    #pragma HLS STREAM variable=spike_out28   depth=16
    #pragma HLS STREAM variable=spike_out29   depth=16
    #pragma HLS STREAM variable=spike_out30   depth=16
    #pragma HLS STREAM variable=spike_out31   depth=16

    // Launch data‑flow processes
    // Note: You may need to define a threshold value for SomaEngine, e.g., params.threshold if available
    SynapseRouter(
        synapse_stream,
        syn_route_in,
        NeuronStart,
        NeuronTotal,
        SimulationTime,
        AmountOfCores,
        syn_forward, syn_forward1, syn_forward2, syn_forward3, syn_forward4, syn_forward5, syn_forward6, syn_forward7,
        syn_forward8, syn_forward9, syn_forward10, syn_forward11, syn_forward12, syn_forward13, syn_forward14, syn_forward15,
        syn_forward16, syn_forward17, syn_forward18, syn_forward19, syn_forward20, syn_forward21, syn_forward22, syn_forward23,
        syn_forward24, syn_forward25, syn_forward26, syn_forward27, syn_forward28, syn_forward29, syn_forward30, syn_forward31,
        syn_forward_rt
    );

    SE0: SomaEngine(threshold, membrane_potential, 0+NeuronStart, MIN(64, NeuronTotal), SimulationTime, syn_forward, spike_out);
    SE1: SomaEngine(threshold, membrane_potential, 64 + NeuronStart, MAX(0, MIN(64, NeuronTotal-64)), SimulationTime, syn_forward1, spike_out1);
    SE2: SomaEngine(threshold, membrane_potential, 128 + NeuronStart, MAX(0, MIN(64, NeuronTotal-128)), SimulationTime, syn_forward2, spike_out2);
    SE3: SomaEngine(threshold, membrane_potential, 192 + NeuronStart, MAX(0, MIN(64, NeuronTotal-192)), SimulationTime, syn_forward3, spike_out3);
    SE4: SomaEngine(threshold, membrane_potential, 256 + NeuronStart, MAX(0, MIN(64, NeuronTotal-256)), SimulationTime, syn_forward4, spike_out4);
    SE5: SomaEngine(threshold, membrane_potential, 320 + NeuronStart, MAX(0, MIN(64, NeuronTotal-320)), SimulationTime, syn_forward5, spike_out5);
    SE6: SomaEngine(threshold, membrane_potential, 384 + NeuronStart, MAX(0, MIN(64, NeuronTotal-384)), SimulationTime, syn_forward6, spike_out6);
    SE7: SomaEngine(threshold, membrane_potential, 448 + NeuronStart, MAX(0, MIN(64, NeuronTotal-448)), SimulationTime, syn_forward7, spike_out7);
    SE8: SomaEngine(threshold, membrane_potential, 512 + NeuronStart, MAX(0, MIN(64, NeuronTotal-512)), SimulationTime, syn_forward8, spike_out8);
    SE9: SomaEngine(threshold, membrane_potential, 576 + NeuronStart, MAX(0, MIN(64, NeuronTotal-576)), SimulationTime, syn_forward9, spike_out9);
    SE10: SomaEngine(threshold, membrane_potential, 640 + NeuronStart, MAX(0, MIN(64, NeuronTotal-640)), SimulationTime, syn_forward10, spike_out10);
    SE11: SomaEngine(threshold, membrane_potential, 704 + NeuronStart, MAX(0, MIN(64, NeuronTotal-704)), SimulationTime, syn_forward11, spike_out11);
    SE12: SomaEngine(threshold, membrane_potential, 768 + NeuronStart, MAX(0, MIN(64, NeuronTotal-768)), SimulationTime, syn_forward12, spike_out12);
    SE13: SomaEngine(threshold, membrane_potential, 832 + NeuronStart, MAX(0, MIN(64, NeuronTotal-832)), SimulationTime, syn_forward13, spike_out13);
    SE14: SomaEngine(threshold, membrane_potential, 896 + NeuronStart, MAX(0, MIN(64, NeuronTotal-896)), SimulationTime, syn_forward14, spike_out14);
    SE15: SomaEngine(threshold, membrane_potential, 960 + NeuronStart, MAX(0, MIN(64, NeuronTotal-960)), SimulationTime, syn_forward15, spike_out15);
    SE16: SomaEngine(threshold, membrane_potential, 1024 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1024)), SimulationTime, syn_forward16, spike_out16);
    SE17: SomaEngine(threshold, membrane_potential, 1088 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1088)), SimulationTime, syn_forward17, spike_out17);
    SE18: SomaEngine(threshold, membrane_potential, 1152 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1152)), SimulationTime, syn_forward18, spike_out18);
    SE19: SomaEngine(threshold, membrane_potential, 1216 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1216)), SimulationTime, syn_forward19, spike_out19);
    SE20: SomaEngine(threshold, membrane_potential, 1280 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1280)), SimulationTime, syn_forward20, spike_out20);
    SE21: SomaEngine(threshold, membrane_potential, 1344 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1344)), SimulationTime, syn_forward21, spike_out21);
    SE22: SomaEngine(threshold, membrane_potential, 1408 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1408)), SimulationTime, syn_forward22, spike_out22);
    SE23: SomaEngine(threshold, membrane_potential, 1472 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1472)), SimulationTime, syn_forward23, spike_out23);
    SE24: SomaEngine(threshold, membrane_potential, 1536 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1536)), SimulationTime, syn_forward24, spike_out24);
    SE25: SomaEngine(threshold, membrane_potential, 1600 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1600)), SimulationTime, syn_forward25, spike_out25);
    SE26: SomaEngine(threshold, membrane_potential, 1664 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1664)), SimulationTime, syn_forward26, spike_out26);
    SE27: SomaEngine(threshold, membrane_potential, 1728 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1728)), SimulationTime, syn_forward27, spike_out27);
    SE28: SomaEngine(threshold, membrane_potential, 1792 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1792)), SimulationTime, syn_forward28, spike_out28);
    SE29: SomaEngine(threshold, membrane_potential, 1856 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1856)), SimulationTime, syn_forward29, spike_out29);
    SE30: SomaEngine(threshold, membrane_potential, 1920 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1920)), SimulationTime, syn_forward30, spike_out30);
    SE31: SomaEngine(threshold, membrane_potential, 1984 + NeuronStart, MAX(0, MIN(64, NeuronTotal-1984)), SimulationTime, syn_forward31, spike_out31);

    accumulate_spike(
        SimulationTime,
        spike_out, spike_out1, spike_out2, spike_out3, spike_out4, spike_out5, spike_out6, spike_out7, 
        spike_out8, spike_out9, spike_out10, spike_out11, spike_out12, spike_out13, spike_out14, spike_out15, 
        spike_out16, spike_out17, spike_out18, spike_out19, spike_out20, spike_out21, spike_out22, spike_out23, 
        spike_out24, spike_out25, spike_out26, spike_out27, spike_out28, spike_out29, spike_out30, spike_out31, 
        spike_stream);
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
