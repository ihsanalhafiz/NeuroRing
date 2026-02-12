#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "hls_half.h"
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1
#define DLY_IDX(neuron, ofs)   ((neuron)*DELAY + (ofs))   // ofs == head[neuron] or (head+delay)


//====================================================================
//  2. AxonLoader – Fetch synapse lists when spikes occur
//====================================================================
extern "C" void NeuroRing(
    ap_uint<256>                 *SpikeRecorder_SynapseList,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     SimulationTime,
    uint32_t                     record_status,
    uint32_t                     CoreID,
    uint32_t                     AmountOfCores,
    hls::stream<stream256u_t>     &SpikeInWeight,
    hls::stream<stream256u_t>    &SynapseStream)
{
    #pragma HLS INTERFACE m_axi port=SpikeRecorder_SynapseList offset=slave bundle=gmem_syn max_widen_bitwidth=256 
    #pragma HLS INTERFACE s_axilite port=SpikeRecorder_SynapseList bundle=control
    #pragma HLS INTERFACE s_axilite port=NeuronStart bundle=control
    #pragma HLS INTERFACE s_axilite port=NeuronTotal bundle=control
    #pragma HLS INTERFACE s_axilite port=SimulationTime bundle=control
    #pragma HLS INTERFACE s_axilite port=record_status bundle=control
    #pragma HLS INTERFACE s_axilite port=CoreID bundle=control
    #pragma HLS INTERFACE s_axilite port=AmountOfCores bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE axis port=SpikeInWeight
    #pragma HLS INTERFACE axis port=SynapseStream

    auto to_core = [&](uint32_t destDelay) -> uint32_t {
        uint32_t global_neuron = (destDelay >> 8) & 0xFFFFFF;        // strip delay
        uint32_t core = global_neuron > 0 ? (uint32_t)((global_neuron - 1)/(NEURON_NUM/8)) : 0xFFF;      // 1-based -> 0-based block, shift by 11 for div by 2048
        return core;
    };

    constexpr uint32_t LOG_STRIDE = NEURON_NUM / 32;   // e.g., 4096/32 = 128

    uint32_t t_temporary = 0;

    const float dt = 0.1f;
    const float tau_m = 10.0f;
    const float tau_syn = 0.5f;
    const float C_m = 250.0f;
    const float E_L = -65.0f;
    const float V_decay = 0.99004983f;   // exp(-dt/tau_m)
    const float I_decay = 0.81873075f;   // exp(-dt/tau_syn)
    const float syn_to_vm = (1.0f/C_m) * ((I_decay - V_decay) / ((1.0f/tau_m) - (1.0f/tau_syn)));
    // new: exact discrete-time gain for a constant bias current
    const float bias_to_vm = (tau_m / C_m) * (1.0f - V_decay);  // mV per pA
    const int   t_ref_steps = 20;        // round(2.0/0.1)

    float U_membPot[NEURON_NUM];   // membrane potential (mV)
    float I_PreSynCurr[NEURON_NUM];
    uint16_t R_RefCnt[NEURON_NUM];
    uint32_t SynapseSize[NEURON_NUM];
    // new: per-neuron DC bias current (pA)
    float I_bias[NEURON_NUM];
    #pragma HLS bind_storage variable=U_membPot   type=ram_2p impl=bram
    #pragma HLS bind_storage variable=I_PreSynCurr type=ram_2p impl=bram
    #pragma HLS bind_storage variable=R_RefCnt    type=ram_2p impl=bram
    #pragma HLS bind_storage variable=SynapseSize type=ram_2p impl=bram
    #pragma HLS bind_storage variable=I_bias      type=ram_2p impl=bram
    #pragma array_partition variable=U_membPot type=block factor=8
    #pragma array_partition variable=I_PreSynCurr type=block factor=8
    #pragma array_partition variable=R_RefCnt type=block factor=8
    #pragma array_partition variable=I_bias type=block factor=8
    
    const float V_th_abs    = -50.0f;
    const float V_th_rel    = V_th_abs    - E_L;
    const float V_reset_abs  = -65.0f;
    const float V_reset_rel  = V_reset_abs - E_L;

    // read parameters from file
    for (int i = 0; i < NeuronTotal; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS loop_tripcount min=4096 max=4096
        int delay_base = i*DELAY;
        int base = (i*SYNAPSE_LIST_SIZE) + SYNAPSE_ARRAY_OFFSET;
        int base256 = base >> 3;
        I_PreSynCurr[i] = 0.0f;
        R_RefCnt[i] = 0;
        ap_uint<256> val_temp;
        val_temp = SpikeRecorder_SynapseList[base256];
        SynapseSize[i] = val_temp.range(31, 0) << 1;
        float_to_uint32 I_bias_temp;
        I_bias_temp.u = val_temp.range(63, 32);
        I_bias[i] = I_bias_temp.f;
        float_to_uint32 UmemPot_temp;
        UmemPot_temp.u = val_temp.range(95, 64);
        U_membPot[i] = UmemPot_temp.f - E_L;
    }

    // Main simulation loop
    read_status_loop: for (int t = 0; t < SimulationTime; t++) {
        //#pragma HLS loop_tripcount min=1000 max=100000
        t_temporary = t;

        uint32_t spike_status[NEURON_NUM/32];
        #pragma HLS array_partition variable=spike_status complete
        init_spike_status: for(int i = 0; i < NEURON_NUM/32; i++) {
            #pragma HLS UNROLL
            spike_status[i] = 0;
        }
        update_membrane_potential: for(int i = 0; i < NEURON_NUM/8; i++) {
            #pragma HLS PIPELINE II=1
            stream256u_t weight_read;
            SpikeInWeight.read(weight_read);
            float_to_uint32 weight_temp;
            weight_temp.u = weight_read.data.range(31, 0);
            float_to_uint32 weight_temp_2;
            weight_temp_2.u = weight_read.data.range(63, 32);
            float_to_uint32 weight_temp_3;
            weight_temp_3.u = weight_read.data.range(95, 64);
            float_to_uint32 weight_temp_4;
            weight_temp_4.u = weight_read.data.range(127, 96);
            float_to_uint32 weight_temp_5;
            weight_temp_5.u = weight_read.data.range(159, 128);
            float_to_uint32 weight_temp_6;
            weight_temp_6.u = weight_read.data.range(191, 160);
            float_to_uint32 weight_temp_7;
            weight_temp_7.u = weight_read.data.range(223, 192);
            float_to_uint32 weight_temp_8;
            weight_temp_8.u = weight_read.data.range(255, 224);
            float weight = weight_temp.f;
            float weight_2 = weight_temp_2.f;
            float weight_3 = weight_temp_3.f;
            float weight_4 = weight_temp_4.f;
            float weight_5 = weight_temp_5.f;
            float weight_6 = weight_temp_6.f;
            float weight_7 = weight_temp_7.f;
            float weight_8 = weight_temp_8.f;
            float v_prev = U_membPot[i];
            float v_prev_2 = U_membPot[i + (NEURON_NUM/8)];
            float v_prev_3 = U_membPot[i + ((NEURON_NUM/8)*2)];
            float v_prev_4 = U_membPot[i + ((NEURON_NUM/8)*3)];
            float v_prev_5 = U_membPot[i + ((NEURON_NUM/8)*4)];
            float v_prev_6 = U_membPot[i + ((NEURON_NUM/8)*5)];
            float v_prev_7 = U_membPot[i + ((NEURON_NUM/8)*6)];
            float v_prev_8 = U_membPot[i + ((NEURON_NUM/8)*7)];
            float i_prev = I_PreSynCurr[i];
            float i_prev_2 = I_PreSynCurr[i + (NEURON_NUM/8)];
            float i_prev_3 = I_PreSynCurr[i + ((NEURON_NUM/8)*2)];
            float i_prev_4 = I_PreSynCurr[i + ((NEURON_NUM/8)*3)];
            float i_prev_5 = I_PreSynCurr[i + ((NEURON_NUM/8)*4)];
            float i_prev_6 = I_PreSynCurr[i + ((NEURON_NUM/8)*5)];
            float i_prev_7 = I_PreSynCurr[i + ((NEURON_NUM/8)*6)];
            float i_prev_8 = I_PreSynCurr[i + ((NEURON_NUM/8)*7)];
            float i_curr = (i_prev * I_decay) + weight;
            float i_curr_2 = (i_prev_2 * I_decay) + weight_2;
            float i_curr_3 = (i_prev_3 * I_decay) + weight_3;
            float i_curr_4 = (i_prev_4 * I_decay) + weight_4;
            float i_curr_5 = (i_prev_5 * I_decay) + weight_5;
            float i_curr_6 = (i_prev_6 * I_decay) + weight_6;
            float i_curr_7 = (i_prev_7 * I_decay) + weight_7;
            float i_curr_8 = (i_prev_8 * I_decay) + weight_8;
            I_PreSynCurr[i] = i_curr;
            I_PreSynCurr[i + (NEURON_NUM/8)] = i_curr_2;
            I_PreSynCurr[i + ((NEURON_NUM/8)*2)] = i_curr_3;
            I_PreSynCurr[i + ((NEURON_NUM/8)*3)] = i_curr_4;
            I_PreSynCurr[i + ((NEURON_NUM/8)*4)] = i_curr_5;
            I_PreSynCurr[i + ((NEURON_NUM/8)*5)] = i_curr_6;
            I_PreSynCurr[i + ((NEURON_NUM/8)*6)] = i_curr_7;
            I_PreSynCurr[i + ((NEURON_NUM/8)*7)] = i_curr_8;

            uint16_t r_prev = R_RefCnt[i];
            uint16_t r_prev_2 = R_RefCnt[i + (NEURON_NUM/8)];
            uint16_t r_prev_3 = R_RefCnt[i + ((NEURON_NUM/8)*2)];
            uint16_t r_prev_4 = R_RefCnt[i + ((NEURON_NUM/8)*3)];
            uint16_t r_prev_5 = R_RefCnt[i + ((NEURON_NUM/8)*4)];
            uint16_t r_prev_6 = R_RefCnt[i + ((NEURON_NUM/8)*5)];
            uint16_t r_prev_7 = R_RefCnt[i + ((NEURON_NUM/8)*6)];
            uint16_t r_prev_8 = R_RefCnt[i + ((NEURON_NUM/8)*7)];
            uint16_t r_next = 0;
            uint16_t r_next_2 = 0;
            uint16_t r_next_3 = 0;
            uint16_t r_next_4 = 0;
            uint16_t r_next_5 = 0;
            uint16_t r_next_6 = 0;
            uint16_t r_next_7 = 0;
            uint16_t r_next_8 = 0;
            float v_candidate = V_reset_rel;
            float v_candidate_2 = V_reset_rel;
            float v_candidate_3 = V_reset_rel;
            float v_candidate_4 = V_reset_rel;
            float v_candidate_5 = V_reset_rel;
            float v_candidate_6 = V_reset_rel;
            float v_candidate_7 = V_reset_rel;
            float v_candidate_8 = V_reset_rel;
            bool spk = false;
            bool spk_2 = false;
            bool spk_3 = false;
            bool spk_4 = false;
            bool spk_5 = false;
            bool spk_6 = false;
            bool spk_7 = false;
            bool spk_8 = false;

            if (r_prev > 0) {
                r_next = (uint16_t)(r_prev - 1);
                v_candidate = V_reset_rel;
            } else {
                float v_new = (v_prev * V_decay) + (i_curr * syn_to_vm) + (I_bias[i] * bias_to_vm);
                spk = (v_new >= V_th_rel);
                v_candidate = spk ? V_reset_rel : v_new;
                r_next = spk ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_2 > 0) {
                r_next_2 = (uint16_t)(r_prev_2 - 1);
                v_candidate_2 = V_reset_rel;
            } else {
                float v_new_2 = (v_prev_2 * V_decay) + (i_curr_2 * syn_to_vm) + (I_bias[i + (NEURON_NUM/8)] * bias_to_vm);
                spk_2 = (v_new_2 >= V_th_rel);
                v_candidate_2 = spk_2 ? V_reset_rel : v_new_2;
                r_next_2 = spk_2 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_3 > 0) {
                r_next_3 = (uint16_t)(r_prev_3 - 1);
                v_candidate_3 = V_reset_rel;
            } else {
                float v_new_3 = (v_prev_3 * V_decay) + (i_curr_3 * syn_to_vm) + (I_bias[i + ((NEURON_NUM/8)*2)] * bias_to_vm);
                spk_3 = (v_new_3 >= V_th_rel);
                v_candidate_3 = spk_3 ? V_reset_rel : v_new_3;
                r_next_3 = spk_3 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_4 > 0) {
                r_next_4 = (uint16_t)(r_prev_4 - 1);
                v_candidate_4 = V_reset_rel;
            } else {
                float v_new_4 = (v_prev_4 * V_decay) + (i_curr_4 * syn_to_vm) + (I_bias[i + ((NEURON_NUM/8)*3)] * bias_to_vm);
                spk_4 = (v_new_4 >= V_th_rel);
                v_candidate_4 = spk_4 ? V_reset_rel : v_new_4;
                r_next_4 = spk_4 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_5 > 0) {
                r_next_5 = (uint16_t)(r_prev_5 - 1);
                v_candidate_5 = V_reset_rel;
            } else {
                float v_new_5 = (v_prev_5 * V_decay) + (i_curr_5 * syn_to_vm) + (I_bias[i + ((NEURON_NUM/8)*4)] * bias_to_vm);
                spk_5 = (v_new_5 >= V_th_rel);
                v_candidate_5 = spk_5 ? V_reset_rel : v_new_5;
                r_next_5 = spk_5 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_6 > 0) {
                r_next_6 = (uint16_t)(r_prev_6 - 1);
                v_candidate_6 = V_reset_rel;
            } else {
                float v_new_6 = (v_prev_6 * V_decay) + (i_curr_6 * syn_to_vm) + (I_bias[i + ((NEURON_NUM/8)*5)] * bias_to_vm);
                spk_6 = (v_new_6 >= V_th_rel);
                v_candidate_6 = spk_6 ? V_reset_rel : v_new_6;
                r_next_6 = spk_6 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_7 > 0) {
                r_next_7 = (uint16_t)(r_prev_7 - 1);
                v_candidate_7 = V_reset_rel;
            } else {
                float v_new_7 = (v_prev_7 * V_decay) + (i_curr_7 * syn_to_vm) + (I_bias[i + ((NEURON_NUM/8)*6)] * bias_to_vm);
                spk_7 = (v_new_7 >= V_th_rel);
                v_candidate_7 = spk_7 ? V_reset_rel : v_new_7;
                r_next_7 = spk_7 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }
            if (r_prev_8 > 0) {
                r_next_8 = (uint16_t)(r_prev_8 - 1);
                v_candidate_8 = V_reset_rel;
            } else {
                float v_new_8 = (v_prev_8 * V_decay) + (i_curr_8 * syn_to_vm) + (I_bias[i + ((NEURON_NUM/8)*7)] * bias_to_vm);
                spk_8 = (v_new_8 >= V_th_rel);
                v_candidate_8 = spk_8 ? V_reset_rel : v_new_8;
                r_next_8 = spk_8 ? (uint16_t)t_ref_steps : (uint16_t)0;
            }

            spike_status[i/32] = spike_status[i/32] | (spk << (i % 32));
            spike_status[(i + (NEURON_NUM/8))/32] = spike_status[(i + (NEURON_NUM/8))/32] | (spk_2 << (i % 32));
            spike_status[(i + ((NEURON_NUM/8)*2))/32] = spike_status[(i + ((NEURON_NUM/8)*2))/32] | (spk_3 << (i % 32));
            spike_status[(i + ((NEURON_NUM/8)*3))/32] = spike_status[(i + ((NEURON_NUM/8)*3))/32] | (spk_4 << (i % 32));
            spike_status[(i + ((NEURON_NUM/8)*4))/32] = spike_status[(i + ((NEURON_NUM/8)*4))/32] | (spk_5 << (i % 32));
            spike_status[(i + ((NEURON_NUM/8)*5))/32] = spike_status[(i + ((NEURON_NUM/8)*5))/32] | (spk_6 << (i % 32));
            spike_status[(i + ((NEURON_NUM/8)*6))/32] = spike_status[(i + ((NEURON_NUM/8)*6))/32] | (spk_7 << (i % 32));
            spike_status[(i + ((NEURON_NUM/8)*7))/32] = spike_status[(i + ((NEURON_NUM/8)*7))/32] | (spk_8 << (i % 32));

            U_membPot[i] = v_candidate;
            R_RefCnt[i] = r_next;
            U_membPot[i + (NEURON_NUM/8)] = v_candidate_2;
            R_RefCnt[i + (NEURON_NUM/8)] = r_next_2;
            U_membPot[i + ((NEURON_NUM/8)*2)] = v_candidate_3;
            R_RefCnt[i + ((NEURON_NUM/8)*2)] = r_next_3;
            U_membPot[i + ((NEURON_NUM/8)*3)] = v_candidate_4;
            R_RefCnt[i + ((NEURON_NUM/8)*3)] = r_next_4;
            U_membPot[i + ((NEURON_NUM/8)*4)] = v_candidate_5;
            R_RefCnt[i + ((NEURON_NUM/8)*4)] = r_next_5;
            U_membPot[i + ((NEURON_NUM/8)*5)] = v_candidate_6;
            R_RefCnt[i + ((NEURON_NUM/8)*5)] = r_next_6;
            U_membPot[i + ((NEURON_NUM/8)*6)] = v_candidate_7;
            R_RefCnt[i + ((NEURON_NUM/8)*6)] = r_next_7;
            U_membPot[i + ((NEURON_NUM/8)*7)] = v_candidate_8;
            R_RefCnt[i + ((NEURON_NUM/8)*7)] = r_next_8;
        }

        // Create active group mask
        ap_uint<NEURON_NUM/32> active_groups = 0;
        check_active: for(int i=0; i<NEURON_NUM/32; i++) {
            #pragma HLS UNROLL
            if(spike_status[i] != 0) active_groups[i] = 1;
        }

        // Processing Spikes - Hierarchical Skip
        process_spikes_outer: while(active_groups != 0) {
            
            // Find active group index
            int i = 0;
            // Handle bits 0-63
            uint64_t lower = active_groups.range(63, 0);
            if (lower != 0) {
                i = __builtin_ctzll(lower);
            } else {
                // Handle remaining bits (for NEURON_NUM=2816, bits 64-87)
                // This generic fallback works for size > 64
                // casting to ensure safe CTZ if needed, though 'else' guarantees non-zero here if loop condition holds
                // logic: if (lower == 0) and (active_groups != 0), then upper must be non-zero.
                 uint64_t upper = active_groups.range((NEURON_NUM/32)-1, 64);
                 i = 64 + __builtin_ctzll(upper);
            }

            // Clear the group bit
            active_groups[i] = 0;

            uint32_t current_status = spike_status[i];
            
            // Inner loop: Process neurons in this group (Already Optimized)
            process_spikes_inner: while (current_status != 0) {
                int j = __builtin_ctz(current_status);
                current_status &= ~(1 << j);

                uint32_t neuron_idx = i*32 + j;
                uint32_t current_synapse_count = SynapseSize[neuron_idx];
                uint32_t start_offset = (neuron_idx * SYNAPSE_LIST_SIZE) + SYNAPSE_ARRAY_OFFSET + 8;
                
                // Burst loop
                fetch_synapses: for(int k = 0; k < current_synapse_count; k+=8) {
                    #pragma HLS PIPELINE II=1
                    uint32_t idx = (start_offset + k) >> 3;
                    ap_uint<256> val_temp = SpikeRecorder_SynapseList[idx];
                    val_temp.range(31, 30) = 0;
                    stream256u_t pkt_out;
                    pkt_out.data = val_temp;
                    SynapseStream.write(pkt_out);
                }
                stream256u_t pkt_out_sync;
                pkt_out_sync.data = 0;
                pkt_out_sync.data.range(31, 31) = 0;
                pkt_out_sync.data.range(30, 30) = 1;
                pkt_out_sync.data.range(29, 8) = CoreID;
                SynapseStream.write(pkt_out_sync);
            }
        }

        stream256u_t sync_packet;
        sync_packet.data = 0;
        sync_packet.data.range(31, 31) = 1;
        sync_packet.data.range(30, 30) = 0;
        sync_packet.data.range(29, 8) = CoreID;
        SynapseStream.write(sync_packet);

        if(record_status == 1) {
            //#pragma HLS DEPENDENCE variable=SpikeRecorder_SynapseList inter false
            write_spike_status: for(int i = 0; i < NEURON_NUM/32; i+=8) {
                #pragma HLS PIPELINE II=1
                uint32_t idx = (t * LOG_STRIDE) + i;
                uint32_t idx256 = idx >> 3;
                ap_uint<256> val_temp;
                val_temp.range(31, 0) = spike_status[i];
                val_temp.range(63, 32) = spike_status[i+1];
                val_temp.range(95, 64) = spike_status[i+2];
                val_temp.range(127, 96) = spike_status[i+3];
                val_temp.range(159, 128) = spike_status[i+4];
                val_temp.range(191, 160) = spike_status[i+5];
                val_temp.range(223, 192) = spike_status[i+6];
                val_temp.range(255, 224) = spike_status[i+7];
                SpikeRecorder_SynapseList[idx256] = val_temp;
            }
        }
    }
    for(int i = 0; i < NEURON_NUM; i+=8) {
        stream256u_t pkt_out;
        SpikeInWeight.read(pkt_out);
    }
}

