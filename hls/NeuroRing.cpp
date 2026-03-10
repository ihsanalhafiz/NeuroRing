#include "NeuroRing.h"
#include "hls_half.h"
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>

#define _XF_SYNTHESIS_ 1
#define DLY_IDX(neuron, ofs)                                                   \
  ((neuron) * DELAY + (ofs)) // ofs == head[neuron] or (head+delay)

//====================================================================
//  2. AxonLoader – Fetch synapse lists when spikes occur
//====================================================================
extern "C" void NeuroRing(ap_uint<256> *SpikeRecorder_SynapseList,
                          uint32_t NeuronStart, uint32_t NeuronTotal, uint32_t SimulationTimeStart,
                          uint32_t SimulationTimeEnd, uint32_t record_status,
                          uint32_t CoreID, uint32_t AmountOfCores,
                          uint32_t V_decay, uint32_t I_decay, uint32_t syn_to_vm,
                          uint32_t bias_to_vm, uint32_t V_th_rel, uint32_t V_reset_rel,
                          uint32_t E_L, uint32_t t_ref_steps,
                          hls::stream<stream256u_t> &SpikeInWeight,
                          hls::stream<stream256u_t> &SynapseStreamRight,
                          hls::stream<stream256u_t> &SynapseStreamLeft) {
#pragma HLS INTERFACE m_axi port = SpikeRecorder_SynapseList offset =          \
    slave bundle = gmem_syn max_widen_bitwidth = 256
#pragma HLS INTERFACE s_axilite port = SpikeRecorder_SynapseList bundle =      \
    control
#pragma HLS INTERFACE s_axilite port = NeuronStart bundle = control
#pragma HLS INTERFACE s_axilite port = NeuronTotal bundle = control
#pragma HLS INTERFACE s_axilite port = SimulationTimeStart bundle = control
#pragma HLS INTERFACE s_axilite port = SimulationTimeEnd bundle = control
#pragma HLS INTERFACE s_axilite port = record_status bundle = control
#pragma HLS INTERFACE s_axilite port = CoreID bundle = control
#pragma HLS INTERFACE s_axilite port = AmountOfCores bundle = control
#pragma HLS INTERFACE s_axilite port = V_decay bundle = control
#pragma HLS INTERFACE s_axilite port = I_decay bundle = control
#pragma HLS INTERFACE s_axilite port = syn_to_vm bundle = control
#pragma HLS INTERFACE s_axilite port = bias_to_vm bundle = control
#pragma HLS INTERFACE s_axilite port = V_th_rel bundle = control
#pragma HLS INTERFACE s_axilite port = V_reset_rel bundle = control
#pragma HLS INTERFACE s_axilite port = E_L bundle = control
#pragma HLS INTERFACE s_axilite port = t_ref_steps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS INTERFACE axis port = SpikeInWeight
#pragma HLS INTERFACE axis port = SynapseStreamRight
#pragma HLS INTERFACE axis port = SynapseStreamLeft

  auto to_core = [&](uint32_t destDelay) -> uint32_t {
    uint32_t global_neuron = (destDelay >> 8) & 0xFFFFFF; // strip delay
    uint32_t core =
        global_neuron > 0
            ? (uint32_t)((global_neuron - 1) / (NEURON_NUM / 8))
            : 0xFFF; // 1-based -> 0-based block, shift by 11 for div by 2048
    return core;
  };

  float U_membPot[NEURON_NUM]; // membrane potential (mV)
  float I_PreSynCurr[NEURON_NUM];
  uint16_t R_RefCnt[NEURON_NUM];
  uint32_t SynapseSize[NEURON_NUM];
  // new: per-neuron DC bias current (pA)
  float I_bias[NEURON_NUM];
#pragma HLS bind_storage variable = U_membPot type = ram_2p impl = bram
#pragma HLS bind_storage variable = I_PreSynCurr type = ram_2p impl = bram
#pragma HLS bind_storage variable = R_RefCnt type = ram_2p impl = bram
#pragma HLS bind_storage variable = SynapseSize type = ram_2p impl = bram
#pragma HLS bind_storage variable = I_bias type = ram_2p impl = bram
#pragma array_partition variable = U_membPot type = block factor = 8
#pragma array_partition variable = I_PreSynCurr type = block factor = 8
#pragma array_partition variable = R_RefCnt type = block factor = 8
#pragma array_partition variable = I_bias type = block factor = 8

  float_to_uint32 V_decay_temp;
  float_to_uint32 I_decay_temp;
  float_to_uint32 syn_to_vm_temp;
  float_to_uint32 bias_to_vm_temp;
  float_to_uint32 V_th_rel_temp;
  float_to_uint32 V_reset_rel_temp;
  float_to_uint32 E_L_temp;
  uint16_t t_ref_steps_temp;
  V_decay_temp.u = V_decay;
  I_decay_temp.u = I_decay;
  syn_to_vm_temp.u = syn_to_vm;
  bias_to_vm_temp.u = bias_to_vm;
  V_th_rel_temp.u = V_th_rel;
  V_reset_rel_temp.u = V_reset_rel;
  E_L_temp.u = E_L;
  t_ref_steps_temp = (uint16_t)t_ref_steps;

  // read parameters from file
  for (int i = 0; i < NeuronTotal; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 4096 max = 4096
    int base = (i * SYNAPSE_LIST_SIZE) + SYNAPSE_ARRAY_OFFSET;
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
read_status_loop:
  for (int t = SimulationTimeStart; t < SimulationTimeEnd; t++) {
    // #pragma HLS loop_tripcount min=1000 max=100000

    uint32_t spike_status[NEURON_NUM / 32];
#pragma HLS array_partition variable = spike_status complete
  init_spike_status:
    for (int i = 0; i < NEURON_NUM / 32; i++) {
#pragma HLS UNROLL
      spike_status[i] = 0;
    }

  update_membrane_potential:
    for (int i = 0; i < NEURON_NUM / 8; i++) {
#pragma HLS PIPELINE II = 1
      stream256u_t weight_read;
      float_to_uint32 weight_temp[8];
      uint16_t r_next[8];
      float v_candidate[8];
      bool spk[8];

      SpikeInWeight.read(weight_read);

      for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
        weight_temp[j].u = weight_read.data.range((j + 1) * 32 - 1, j * 32);
        I_PreSynCurr[i + ((NEURON_NUM / 8) * j)] =
            (I_PreSynCurr[i + ((NEURON_NUM / 8) * j)] * I_decay_temp.f) +
            weight_temp[j].f;
        if (R_RefCnt[i + ((NEURON_NUM / 8) * j)] > 0) {
          r_next[j] = (uint16_t)(R_RefCnt[i + ((NEURON_NUM / 8) * j)] - 1);
          v_candidate[j] = V_reset_rel_temp.f;
        } else {
          float v_new =
              (U_membPot[i + ((NEURON_NUM / 8) * j)] * V_decay_temp.f) +
              (I_PreSynCurr[i + ((NEURON_NUM / 8) * j)] * syn_to_vm_temp.f) +
              (I_bias[i + ((NEURON_NUM / 8) * j)] * bias_to_vm_temp.f);
          spk[j] = (v_new >= V_th_rel_temp.f);
          v_candidate[j] = spk[j] ? V_reset_rel_temp.f : v_new;
          r_next[j] = spk[j] ? t_ref_steps_temp : (uint16_t)0;
        }
        spike_status[(i + ((NEURON_NUM / 8) * j)) / 32] =
            spike_status[(i + ((NEURON_NUM / 8) * j)) / 32] |
            (spk[j] << (i % 32));
        U_membPot[i + ((NEURON_NUM / 8) * j)] = v_candidate[j];
        R_RefCnt[i + ((NEURON_NUM / 8) * j)] = r_next[j];
      }
    }

    for (int i = 0; i < NEURON_NUM / 32; i++) {
      if (spike_status[i] != 0) {
        for (int j = 0; j < 32; j++) {
          if (((spike_status[i] >> j) & 1) != 0) {
            uint32_t current_synapse_count = SynapseSize[i * 32 + j];
            uint32_t start_offset =
                ((i * 32 + j) * SYNAPSE_LIST_SIZE) + SYNAPSE_ARRAY_OFFSET + 8;
            for (int k = 0; k < current_synapse_count; k += 8) {
#pragma HLS PIPELINE II = 1 rewind
              uint32_t idx = (start_offset + k) >> 3;
              ap_uint<256> val_temp = SpikeRecorder_SynapseList[idx];
              stream256u_t pkt_out;
              pkt_out.data = val_temp;
              uint32_t dest = to_core(val_temp.range(31, 0));
              uint32_t d_r = (dest - (CoreID * 8) + (AmountOfCores * 8)) %
                             (AmountOfCores * 8);
              uint32_t d_l = ((CoreID * 8) - dest + (AmountOfCores * 8)) %
                             (AmountOfCores * 8);
              if (d_l < d_r) {
                SynapseStreamLeft.write(pkt_out);
              } else {
                SynapseStreamRight.write(pkt_out);
              }
            }
            stream256u_t pkt_out_sync;
            pkt_out_sync.data = 0;
            pkt_out_sync.data.range(30, 30) = 1;
            pkt_out_sync.data.range(29, 8) = CoreID;
            SynapseStreamRight.write(pkt_out_sync);
          }
        }
      }
    }

    stream256u_t sync_packet;
    sync_packet.data = 0;
    sync_packet.data.range(31, 31) = 1;
    sync_packet.data.range(29, 8) = CoreID;
    SynapseStreamRight.write(sync_packet);
    SynapseStreamLeft.write(sync_packet);

    if (record_status == 1) {
    // #pragma HLS DEPENDENCE variable=SpikeRecorder_SynapseList inter false
    write_spike_status:
      for (int i = 0; i < NEURON_NUM / 32; i += 8) {
#pragma HLS PIPELINE II = 1
        uint32_t idx = (t * (NEURON_NUM / 32)) + i;
        uint32_t idx256 = idx >> 3;
        ap_uint<256> val_temp;
        val_temp.range(31, 0) = spike_status[i];
        val_temp.range(63, 32) = spike_status[i + 1];
        val_temp.range(95, 64) = spike_status[i + 2];
        val_temp.range(127, 96) = spike_status[i + 3];
        val_temp.range(159, 128) = spike_status[i + 4];
        val_temp.range(191, 160) = spike_status[i + 5];
        val_temp.range(223, 192) = spike_status[i + 6];
        val_temp.range(255, 224) = spike_status[i + 7];
        SpikeRecorder_SynapseList[idx256] = val_temp;
      }
    }
  }

  for (int i = 0; i < NeuronTotal; i++) {
    #pragma HLS PIPELINE II = 1
    #pragma HLS loop_tripcount min = 4096 max = 4096
        int base = (i * SYNAPSE_LIST_SIZE) + SYNAPSE_ARRAY_OFFSET;
        int base256 = base >> 3;
        ap_uint<256> val_temp;
        val_temp.range(31, 0) = SynapseSize[i] >> 1;
        float_to_uint32 I_bias_temp;
        I_bias_temp.f = I_bias[i];
        val_temp.range(63, 32) = I_bias_temp.u;
        float_to_uint32 UmemPot_temp;
        UmemPot_temp.f = U_membPot[i] + E_L_temp.f;
        val_temp.range(95, 64) = UmemPot_temp.u;
        SpikeRecorder_SynapseList[base256] = val_temp;
  }
  
  for (int i = 0; i < NEURON_NUM; i += 8) {
    stream256u_t pkt_out;
    SpikeInWeight.read(pkt_out);
  }
}
