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
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine1(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine2(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine3(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine4(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine5(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine6(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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
    }
}

void SomaEngine7(
    uint32_t                     threshold,
    uint32_t                     membrane_potential,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    hls::stream<lanes8_t>  &SpikeStream,
    uint32_t                     SimulationTime,
    hls::stream<stream512u_t>  &SpikeOut)
{
    //----------------------------------------------------------
    // Local spike status memory (2048 neurons ⇒ 64 × 32‑bit)
    //----------------------------------------------------------
    stream512u_t spike_status;
    spike_status.data = 0;

    const float alpha = 0.99;
    const float gamma = 0.00036;
    const float beta = 0.82;
    const float t_ref = 20;
    const float w_f = 585;

    bool runstate = true;

    float U_membPot[NEURON_NUM/8];
    float I_PreSynCurr[NEURON_NUM/8];
    float R_RefCnt[NEURON_NUM/8];
    float x_state[NEURON_NUM/8];
    float C_acc[NEURON_NUM/8];

    float_to_uint32 threshold_conv;
    threshold_conv.u = threshold;
    float threshold_float = threshold_conv.f;

    float_to_uint32 membrane_potential_conv;
    membrane_potential_conv.u = membrane_potential;
    float membrane_potential_float = membrane_potential_conv.f;

    for(int i = 0; i < NEURON_NUM/8 ; i++) {
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
    

        //--------------------------------------------------
        // 2) Consume incoming weighted spikes
        //--------------------------------------------------
        runstate = true;
        synapse_loop: while (runstate) {
            //#pragma HLS PIPELINE II=1
            lanes8_t pkt;
            bool have_pkt = SpikeStream.read_nb(pkt);
            if (have_pkt) {
                DstID_t dst[NLANE];
                float weight[NLANE];
                float_to_uint32 weight_conv[NLANE];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=weight complete
                #pragma HLS ARRAY_PARTITION variable=weight_conv complete
                read_pkt: for (int i = 0; i < NLANE; i++) {
                    #pragma HLS UNROLL
                    weight_conv[i].u = pkt.data[i].range(31, 0);
                    dst[i] = pkt.data[i].range(63, 40);
                    weight[i] = weight_conv[i].f;
                    C_acc[dst[i].to_uint()-NeuronStart] = weight[i];
                }
                //printf("SomaEngine loop, dst: %u, delay: %u, weight: %f\n", dst, (uint32_t)((pkt.data >> 32) & 0xFF), weight_conv.f);
                if (dst[0] == 0xFFFFFF) {
                    // Sync – end of timestep
                    runstate = false;
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

    const uint32_t start0 = NeuronStart;
    const uint32_t start1 = NeuronStart + ((NeuronTotal+7)/8)*1;
    const uint32_t start2 = NeuronStart + ((NeuronTotal+7)/8)*2;
    const uint32_t start3 = NeuronStart + ((NeuronTotal+7)/8)*3;
    const uint32_t start4 = NeuronStart + ((NeuronTotal+7)/8)*4;
    const uint32_t start5 = NeuronStart + ((NeuronTotal+7)/8)*5;
    const uint32_t start6 = NeuronStart + ((NeuronTotal+7)/8)*6;
    const uint32_t start7 = NeuronStart + ((NeuronTotal+7)/8)*7;

    const uint32_t end0 = start1;
    const uint32_t end1 = start2;
    const uint32_t end2 = start3;
    const uint32_t end3 = start4;
    const uint32_t end4 = start5;
    const uint32_t end5 = start6;
    const uint32_t end6 = start7;
    const uint32_t end7 = neuron_end;
    
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
                uint32_t weight_bits[8];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=delay complete
                #pragma HLS ARRAY_PARTITION variable=weight_bits complete
                
                // Unpack all 8 synapses at once
                for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
                    dst[i] = pkt.data.range(base_bit, base_bit - 23);
                    delay[i] = pkt.data.range(base_bit - 24, base_bit - 31);
                    weight_bits[i] = pkt.data.range(base_bit - 32, base_bit - 63);
                }
                
                // Check if this is an axon done signal
                if (delay[0] == 0xFE) {
                    axon_done = true;
                    // Non-blocking write with retry
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SynForwardRoute.write_nb(pkt);
                    }
                } else {
                    // Process all 8 synapses efficiently
                    synapse_loop: for (int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                        // Create synapse word
                        synapse_word_t temp;
                        temp.range(63, 40) = dst[i];
                        temp.range(39, 32) = delay[i];
                        temp.range(31, 0) = weight_bits[i];
                        
                        // Route based on destination
                        bool is_local0 = (dst[i] >= start0 && dst[i] < end0);
                        bool is_local1 = (dst[i] >= start1 && dst[i] < end1);
                        bool is_local2 = (dst[i] >= start2 && dst[i] < end2);
                        bool is_local3 = (dst[i] >= start3 && dst[i] < end3);
                        bool is_local4 = (dst[i] >= start4 && dst[i] < end4);
                        bool is_local5 = (dst[i] >= start5 && dst[i] < end5);
                        bool is_local6 = (dst[i] >= start6 && dst[i] < end6);
                        bool is_local7 = (dst[i] >= start7 && dst[i] < end7);
                        if (is_local0) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local1) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward1.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local2) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward2.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local3) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward3.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local4) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward4.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local5) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward5.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local6) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward6.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local7) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward7.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        }
                    }
                    if((dst[0] != 0) && (dst[1] != 0) && (dst[2] != 0) && (dst[3] != 0) && (dst[4] != 0) && (dst[5] != 0) && (dst[6] != 0) && (dst[7] != 0)) {
                        // create stream512u_t packet
                        stream512u_t temp_pkt;
                        for(int i = 0; i < 8; i++) {
                            #pragma HLS UNROLL
                            int base_bit = 511 - i * 64;
                            temp_pkt.data.range(base_bit, base_bit - 23) = dst[i];
                            temp_pkt.data.range(base_bit - 24, base_bit - 31) = delay[i];
                            temp_pkt.data.range(base_bit - 32, base_bit - 63) = weight_bits[i];
                        }
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForwardRoute.write_nb(temp_pkt);
                        }
                    }
                }
            }
            
            // Process routed stream from previous router
            stream512u_t temp_rt;
            bool have_rt = SynapseStreamRoute.read_nb(temp_rt);
            if (have_rt) {
                DstID_t dst[8];
                Delay_t delay[8];
                uint32_t weight_bits[8];
                #pragma HLS ARRAY_PARTITION variable=dst complete
                #pragma HLS ARRAY_PARTITION variable=delay complete
                #pragma HLS ARRAY_PARTITION variable=weight_bits complete

                for(int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
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
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForwardRoute.write_nb(temp_rt);
                        }
                    }
                } else {
                    // Route based on destination
                    synapse_loop1: for (int i = 0; i < 8; i++) {
                        #pragma HLS UNROLL
                        // Create synapse word
                        synapse_word_t temp;
                        temp.range(63, 40) = dst[i];
                        temp.range(39, 32) = delay[i];
                        temp.range(31, 0) = weight_bits[i];
                        
                        // Route based on destination
                        bool is_local0 = (dst[i] >= start0 && dst[i] < end0);
                        bool is_local1 = (dst[i] >= start1 && dst[i] < end1);
                        bool is_local2 = (dst[i] >= start2 && dst[i] < end2);
                        bool is_local3 = (dst[i] >= start3 && dst[i] < end3);
                        bool is_local4 = (dst[i] >= start4 && dst[i] < end4);
                        bool is_local5 = (dst[i] >= start5 && dst[i] < end5);
                        bool is_local6 = (dst[i] >= start6 && dst[i] < end6);
                        bool is_local7 = (dst[i] >= start7 && dst[i] < end7);
                        if (is_local0) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local1) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward1.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local2) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward2.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local3) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward3.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local4) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward4.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local5) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward5.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local6) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward6.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        } else if (is_local7) {
                            // Write to local synapse stream
                            bool write_status = false;
                            while(!write_status) {
                                write_status = SynForward7.write_nb(temp);
                            }
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        }
                    }
                    if((dst[0] != 0) && (dst[1] != 0) && (dst[2] != 0) && (dst[3] != 0) && (dst[4] != 0) && (dst[5] != 0) && (dst[6] != 0) && (dst[7] != 0)) {
                        // create stream512u_t packet
                        stream512u_t temp_pkt;
                        for(int i = 0; i < 8; i++) {
                            #pragma HLS UNROLL
                            int base_bit = 511 - i * 64;
                            temp_pkt.data.range(base_bit, base_bit - 23) = dst[i];
                            temp_pkt.data.range(base_bit - 24, base_bit - 31) = delay[i];
                            temp_pkt.data.range(base_bit - 32, base_bit - 63) = weight_bits[i];
                        }
                        bool write_status = false;
                        while(!write_status) {
                            write_status = SynForwardRoute.write_nb(temp_pkt);
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
    }
}

//====================================================================
//  4. DendriteDelay – Implements discrete delay line per synapse
//====================================================================
void DendriteDelay(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay1(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay2(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay3(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay4(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay5(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay6(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
                }
            }
        }
    }
}
void DendriteDelay7(
    hls::stream<synapse_word_t> &SynForward,
    uint32_t                     NeuronStart,
    uint32_t                     SimulationTime,
    hls::stream<lanes8_t> &SpikeStream)
{
    //------------------------------------------------------
    //  On‑chip circular buffer to hold delayed packets
    //------------------------------------------------------
    //static hls::stream<synapse_word_t> delay_fifo;
    static float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    #pragma HLS ARRAY_PARTITION variable=buf_flat  dim=1 cyclic factor=NLANE
    #pragma HLS ARRAY_PARTITION variable=head dim=1 cyclic factor=NLANE
    //------------------------------------------------------
    //  Zero-initialize buffer and head pointers
    //------------------------------------------------------
    init_loop_outer: for (int core = 0; core < NCORE; core++) {
        head[core] = 0;
        init_loop_inner: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat[BUF_IDX(core,d)] = 0.0f;
        }
    }

    delay_loop: for (int t = 0; t < SimulationTime; t++) {
        //--------------------------------------------------
        // 1) send the delayed packets and move head pointer
        //--------------------------------------------------
        for (int i = 0; i < GROUP; i++) {
            #pragma HLS PIPELINE II=1 
            lanes8_t pkt;
            #pragma HLS ARRAY_PARTITION variable=pkt.data complete
            for (int lane = 0; lane < NLANE; lane++) {
                #pragma HLS UNROLL
                const int core = i * NLANE + lane;
                ap_uint<6> h = head[core];
                float weight = buf_flat[BUF_IDX(core,h)];
                buf_flat[BUF_IDX(core,h)] = 0.0f;
                float_to_uint32 temp_conv;
                temp_conv.f = weight;
                DstID_t dst = core+NeuronStart;
                synapse_word_t temp;
                temp.range(63, 40) = dst;
                temp.range(39, 32) = 0x0;
                temp.range(31, 0) = temp_conv.u;
                pkt.data[lane] = temp;
                head[core] = (h + 1);
            }
            bool write_status = false;
            while(!write_status) {
                write_status = SpikeStream.write_nb(pkt);
            }
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
                ap_uint<6> h2 = (head[dst.to_uint()-NeuronStart] + delay) & 0x3F;
                float weight2 = buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)];
                if (dst == 0xFFFFFF) {
                    // Sync word – forward immediately & exit timestep
                    lanes8_t pkt;
                    pkt.data[0] = pkt_new;
                    for (int i = 1; i < NLANE; i++) {
                        #pragma HLS UNROLL
                        pkt.data[i] = 0;
                    }
                    bool write_status = false;
                    while(!write_status) {
                        write_status = SpikeStream.write_nb(pkt);
                    }
                    done = true;            // one timestep completed
                } else {
                    buf_flat[BUF_IDX((dst.to_uint()-NeuronStart),h2)] = weight2 + weight;
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
    hls::stream<stream512u_t> &syn_route_in,
    hls::stream<stream512u_t> &syn_forward_rt,
    hls::stream<stream512u_t> &synapse_stream,
    hls::stream<stream512u_t> &spike_out,
    hls::stream<stream512u_t> &spike_out1,
    hls::stream<stream512u_t> &spike_out2,
    hls::stream<stream512u_t> &spike_out3,
    hls::stream<stream512u_t> &spike_out4,
    hls::stream<stream512u_t> &spike_out5,
    hls::stream<stream512u_t> &spike_out6,
    hls::stream<stream512u_t> &spike_out7
)
{
#pragma HLS INTERFACE axis port=synapse_stream bundle=AXIS_IN
#pragma HLS INTERFACE axis port=spike_out bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out1 bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out2 bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out3 bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out4 bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out5 bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out6 bundle=AXIS_OUT
#pragma HLS INTERFACE axis port=spike_out7 bundle=AXIS_OUT
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
    hls::stream<lanes8_t> spike_stream;
    hls::stream<lanes8_t> spike_stream1;
    hls::stream<lanes8_t> spike_stream2;
    hls::stream<lanes8_t> spike_stream3;
    hls::stream<lanes8_t> spike_stream4;
    hls::stream<lanes8_t> spike_stream5;
    hls::stream<lanes8_t> spike_stream6;
    hls::stream<lanes8_t> spike_stream7;

#pragma HLS STREAM variable=syn_forward     depth=256
#pragma HLS STREAM variable=syn_forward1    depth=256
#pragma HLS STREAM variable=syn_forward2    depth=256
#pragma HLS STREAM variable=syn_forward3    depth=256
#pragma HLS STREAM variable=syn_forward4    depth=256
#pragma HLS STREAM variable=syn_forward5    depth=256
#pragma HLS STREAM variable=syn_forward6    depth=256
#pragma HLS STREAM variable=syn_forward7    depth=256
#pragma HLS STREAM variable=spike_stream    depth=256
#pragma HLS STREAM variable=spike_stream1    depth=256
#pragma HLS STREAM variable=spike_stream2    depth=256
#pragma HLS STREAM variable=spike_stream3    depth=256
#pragma HLS STREAM variable=spike_stream4    depth=256
#pragma HLS STREAM variable=spike_stream5    depth=256
#pragma HLS STREAM variable=spike_stream6    depth=256
#pragma HLS STREAM variable=spike_stream7    depth=256

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
        syn_forward1,
        syn_forward2,
        syn_forward3,
        syn_forward4,
        syn_forward5,
        syn_forward6,
        syn_forward7,
        syn_forward_rt
    );

    DendriteDelay(syn_forward, NeuronStart, SimulationTime, spike_stream);   
    DendriteDelay1(syn_forward1, NeuronStart+(((NeuronTotal+7)/8)*1), SimulationTime, spike_stream1);
    DendriteDelay2(syn_forward2, NeuronStart+(((NeuronTotal+7)/8)*2), SimulationTime, spike_stream2);
    DendriteDelay3(syn_forward3, NeuronStart+(((NeuronTotal+7)/8)*3), SimulationTime, spike_stream3);
    DendriteDelay4(syn_forward4, NeuronStart+(((NeuronTotal+7)/8)*4), SimulationTime, spike_stream4);
    DendriteDelay5(syn_forward5, NeuronStart+(((NeuronTotal+7)/8)*5), SimulationTime, spike_stream5);
    DendriteDelay6(syn_forward6, NeuronStart+(((NeuronTotal+7)/8)*6), SimulationTime, spike_stream6);
    DendriteDelay7(syn_forward7, NeuronStart+(((NeuronTotal+7)/8)*7), SimulationTime, spike_stream7);

    SomaEngine(threshold, membrane_potential, 
        NeuronStart, ((NeuronTotal+7)/8), spike_stream, SimulationTime,
        spike_out
    );
    SomaEngine1(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*1), ((NeuronTotal+7)/8), spike_stream1, SimulationTime,
        spike_out1
    );
    SomaEngine2(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*2), ((NeuronTotal+7)/8), spike_stream2, SimulationTime,
        spike_out2
    );
    SomaEngine3(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*3), ((NeuronTotal+7)/8), spike_stream3, SimulationTime,
        spike_out3
    );
    SomaEngine4(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*4), ((NeuronTotal+7)/8), spike_stream4, SimulationTime,
        spike_out4
    );
    SomaEngine5(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*5), ((NeuronTotal+7)/8), spike_stream5, SimulationTime,
        spike_out5
    );
    SomaEngine6(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*6), ((NeuronTotal+7)/8), spike_stream6, SimulationTime,
        spike_out6
    );
    SomaEngine7(threshold, membrane_potential, 
        NeuronStart+(((NeuronTotal+7)/8)*7), NeuronTotal - (((NeuronTotal+7)/8)*7), spike_stream7, SimulationTime,
        spike_out7
    );
}



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
