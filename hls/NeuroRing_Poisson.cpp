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

// 32-bit Galois LFSR RNG (fast, synthesizable)
// static ap_uint<32> lfsr32_next(ap_uint<32> s) {
// #pragma HLS INLINE
//    // Polynomial/taps example: x^32 + x^22 + x^2 + x + 1
//    ap_uint<1> lsb = s[0];
//    s >>= 1;
//    if (lsb) s ^= 0x80200003u;
//    return s;
//}

static ap_uint<32> xorshift32_next(ap_uint<32> s) {
#pragma HLS INLINE
  // A standard, highly-tested Xorshift triplet
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

//====================================================================
//  2. Poisson – Fetch synapse lists when spikes occur
//====================================================================
extern "C" void
NeuroRing_Poisson(ap_uint<256> *SpikeRecorder_SynapseList, uint32_t NeuronStart,
                  uint32_t NeuronTotal, uint32_t SimulationTime,
                  uint32_t record_status, uint32_t CoreID,
                  uint32_t AmountOfCores,
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

  uint32_t SeedNumber[NEURON_NUM];
  uint32_t PoissonProb[NEURON_NUM];
  uint32_t SynapseSize[NEURON_NUM];
  uint32_t rnd_prev[NEURON_NUM];
  uint32_t rnd_curr[NEURON_NUM];

#pragma HLS bind_storage variable = SeedNumber type = ram_2p impl = bram
#pragma HLS bind_storage variable = PoissonProb type = ram_2p impl = bram
#pragma HLS bind_storage variable = SynapseSize type = ram_2p impl = bram
#pragma HLS bind_storage variable = rnd_prev type = ram_2p impl = bram
#pragma HLS bind_storage variable = rnd_curr type = ram_2p impl = bram

#pragma array_partition variable = SeedNumber type = block factor = 8
#pragma array_partition variable = PoissonProb type = block factor = 8
#pragma array_partition variable = SynapseSize type = block factor = 8
#pragma array_partition variable = rnd_prev type = block factor = 8
#pragma array_partition variable = rnd_curr type = block factor = 8

  // read parameters from file
  for (int i = 0; i < NeuronTotal; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 4096 max = 4096
    int delay_base = i * DELAY;
    int base = (i * SYNAPSE_LIST_SIZE) + SYNAPSE_ARRAY_OFFSET;
    int base256 = base >> 3;
    ap_uint<256> val_temp;
    val_temp = SpikeRecorder_SynapseList[base256];
    SynapseSize[i] = val_temp.range(31, 0) << 1;
    SeedNumber[i] = val_temp.range(63, 32);
    PoissonProb[i] = val_temp.range(95, 64);
    rnd_prev[i] = xorshift32_next(SeedNumber[i]);
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
        uint32_t idx = i + ((NEURON_NUM / 8) * j);
        rnd_curr[idx] = xorshift32_next(rnd_prev[idx]);
        rnd_prev[idx] = rnd_curr[idx];
        if (idx < NeuronTotal) {
          spk[j] = (rnd_curr[idx] < PoissonProb[idx]);
        } else {
          spk[j] = 0;
        }
        spike_status[idx / 32] = spike_status[idx / 32] | (spk[j] << (i % 32));
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
