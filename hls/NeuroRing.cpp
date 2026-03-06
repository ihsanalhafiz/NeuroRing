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
                          uint32_t NeuronStart, uint32_t NeuronTotal,
                          uint32_t SimulationTime, uint32_t record_status,
                          uint32_t CoreID, uint32_t AmountOfCores,
                          float V_decay, float I_decay, float syn_to_vm,
                          float bias_to_vm, float V_th_rel, float V_reset_rel,
                          float E_L, uint32_t t_ref_steps,
                          hls::stream<stream256u_t> &SpikeInWeight,
                          hls::stream<stream256u_t> &SynapseStreamRight,
                          hls::stream<stream256u_t> &SynapseStreamLeft) {
#pragma HLS INTERFACE m_axi port = SpikeRecorder_SynapseList offset =          \
    slave bundle = gmem_syn max_widen_bitwidth = 256
#pragma HLS INTERFACE s_axilite port = SpikeRecorder_SynapseList bundle =      \
    control
#pragma HLS INTERFACE s_axilite port = NeuronStart bundle = control
#pragma HLS INTERFACE s_axilite port = NeuronTotal bundle = control
#pragma HLS INTERFACE s_axilite port = SimulationTime bundle = control
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

  // const float dt = 0.1f;
  // const float tau_m = 10.0f;
  // const float tau_syn = 0.5f;
  // const float C_m = 250.0f;
  // const float E_L = -65.0f;
  // const float V_decay = 0.99004983f;   // exp(-dt/tau_m)
  // const float I_decay = 0.81873075f;   // exp(-dt/tau_syn)
  // const float syn_to_vm = (1.0f/C_m) * ((I_decay - V_decay) / ((1.0f/tau_m) -
  // (1.0f/tau_syn)));
  //// new: exact discrete-time gain for a constant bias current
  // const float bias_to_vm = (tau_m / C_m) * (1.0f - V_decay);  // mV per pA
  // const int   t_ref_steps = 20;        // round(2.0/0.1)

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

  // const float V_th_abs    = -50.0f;
  // const float V_th_rel    = V_th_abs    - E_L;
  // const float V_reset_abs  = -65.0f;
  // const float V_reset_rel  = V_reset_abs - E_L;

  // read parameters from file
  for (int i = 0; i < NeuronTotal; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 4096 max = 4096
    int delay_base = i * DELAY;
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
    U_membPot[i] = UmemPot_temp.f + E_L;
  }

// Main simulation loop
read_status_loop:
  for (int t = 0; t < SimulationTime; t++) {
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
            (I_PreSynCurr[i + ((NEURON_NUM / 8) * j)] * I_decay) +
            weight_temp[j].f;
        if (R_RefCnt[i + ((NEURON_NUM / 8) * j)] > 0) {
          r_next[j] = (uint16_t)(R_RefCnt[i + ((NEURON_NUM / 8) * j)] - 1);
          v_candidate[j] = V_reset_rel;
        } else {
          float v_new_3 =
              (U_membPot[i + ((NEURON_NUM / 8) * j)] * V_decay) +
              (I_PreSynCurr[i + ((NEURON_NUM / 8) * j)] * syn_to_vm) +
              (I_bias[i + ((NEURON_NUM / 8) * j)] * bias_to_vm);
          spk[j] = (v_new_3 >= V_th_rel);
          v_candidate[j] = spk[j] ? V_reset_rel : v_new_3;
          r_next[j] = spk[j] ? (uint16_t)t_ref_steps : (uint16_t)0;
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
  for (int i = 0; i < NEURON_NUM; i += 8) {
    stream256u_t pkt_out;
    SpikeInWeight.read(pkt_out);
  }
}
