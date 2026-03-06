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

#include "NeuroRing.h"
#include "hls_half.h"
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>

#define BUF_IDX(core, ofs)                                                     \
  ((core) * DELAY + (ofs)) // ofs == head[core] or (head+delay)

//====================================================================
//  3. SynapseRouter – Route packets to local or next core
//====================================================================
void RouterRight(hls::stream<stream256u_t> &SynapseStream,
                 hls::stream<stream256u_t> &SynapseStreamRoute,
                 uint32_t SimulationTime, uint32_t AmountOfCores,
                 uint32_t CoreID, hls::stream<synapse_list_t> &SynForwardOut0,
                 hls::stream<synapse_list_t> &SynForwardOut1,
                 hls::stream<synapse_list_t> &SynForwardOut2,
                 hls::stream<synapse_list_t> &SynForwardOut3,
                 hls::stream<synapse_list_t> &SynForwardOut4,
                 hls::stream<synapse_list_t> &SynForwardOut5,
                 hls::stream<synapse_list_t> &SynForwardOut6,
                 hls::stream<synapse_list_t> &SynForwardOut7,
                 hls::stream<stream256u_t> &SynForwardRoute) {
  auto to_core = [&](uint32_t destDelay) -> uint32_t {
    uint32_t global_neuron = (destDelay >> 8) & 0xFFFFFF; // strip delay
    uint32_t core =
        global_neuron > 0
            ? (uint32_t)((global_neuron - 1) / (NEURON_NUM / 8))
            : 0xFFF; // 1-based -> 0-based block, shift by 11 for div by 2048
    return core;
  };

  // Main routing loop over simulation time

router_loop:
  for (int t = 0; t < SimulationTime; t++) {
    bool prev_done = false;
    bool read_axon = true;
    uint32_t coreDone = 0;

    while (!(prev_done)) {
#pragma HLS PIPELINE II = 1 rewind

      // Process main synapse stream
      stream256u_t pkt;
      bool have_pkt = false;
      bool read_axonLoader = false;

      if (SynapseStreamRoute.read_nb(pkt)) {
        have_pkt = true;
        read_axonLoader = false;
      } else if (read_axon) {
        if (SynapseStream.read_nb(pkt)) {
          have_pkt = true;
          read_axonLoader = true;
        }
      }
      if (have_pkt) {
        if (pkt.data[31] == 1) {
          if (read_axonLoader == true) { // read from axonLoader
            SynForwardRoute.write(pkt);
          } else { // read from route stream
            if (coreDone == AmountOfCores - 1) {
              prev_done = true;
            } else {
              coreDone++;
            }

            // Forward if not for this core
            if ((pkt.data.range(29, 8)) != CoreID) {
              SynForwardRoute.write(pkt);
            }
          }
        } else {
          if (pkt.data[30] == 1) {
            if (read_axonLoader == false) {
              if (pkt.data.range(29, 8) == CoreID) {
                read_axon = true;
              } else {
                SynForwardRoute.write(pkt);
              }
            } else { // token came from local AxonLoader (SynapseStream)
              SynForwardRoute.write(pkt);
              read_axon = false;
            }
          } else {
            uint32_t dest = to_core(pkt.data.range(29, 0));
            uint32_t dest2 = to_core(pkt.data.range(221, 192));
            if ((dest2 / 8) != CoreID && dest2 != 0xFFF) {
              SynForwardRoute.write(pkt);
            }
            if ((dest / 8) == CoreID || (dest2 / 8) == CoreID) {
              uint32_t lane0 = (dest / 8) == CoreID ? (dest % 8) : 0xFFFFFFFF;
              uint32_t lane1 = (dest2 / 8) == CoreID ? (dest2 % 8) : 0xFFFFFFFF;
              if (lane0 == 0 || lane1 == 0)
                SynForwardOut0.write(pkt.data);
              if (lane0 == 1 || lane1 == 1)
                SynForwardOut1.write(pkt.data);
              if (lane0 == 2 || lane1 == 2)
                SynForwardOut2.write(pkt.data);
              if (lane0 == 3 || lane1 == 3)
                SynForwardOut3.write(pkt.data);
              if (lane0 == 4 || lane1 == 4)
                SynForwardOut4.write(pkt.data);
              if (lane0 == 5 || lane1 == 5)
                SynForwardOut5.write(pkt.data);
              if (lane0 == 6 || lane1 == 6)
                SynForwardOut6.write(pkt.data);
              if (lane0 == 7 || lane1 == 7)
                SynForwardOut7.write(pkt.data);
            }
          }
        }
      }
    }

    synapse_list_t pkt_sync;
    pkt_sync = 0;
    SynForwardOut0.write(pkt_sync);
    SynForwardOut1.write(pkt_sync);
    SynForwardOut2.write(pkt_sync);
    SynForwardOut3.write(pkt_sync);
    SynForwardOut4.write(pkt_sync);
    SynForwardOut5.write(pkt_sync);
    SynForwardOut6.write(pkt_sync);
    SynForwardOut7.write(pkt_sync);
  }
}

void RouterLeft(hls::stream<stream256u_t> &SynapseStream,
                hls::stream<stream256u_t> &SynapseStreamRoute,
                uint32_t SimulationTime, uint32_t AmountOfCores,
                uint32_t CoreID, hls::stream<synapse_list_t> &SynForwardOut0,
                hls::stream<synapse_list_t> &SynForwardOut1,
                hls::stream<synapse_list_t> &SynForwardOut2,
                hls::stream<synapse_list_t> &SynForwardOut3,
                hls::stream<synapse_list_t> &SynForwardOut4,
                hls::stream<synapse_list_t> &SynForwardOut5,
                hls::stream<synapse_list_t> &SynForwardOut6,
                hls::stream<synapse_list_t> &SynForwardOut7,
                hls::stream<stream256u_t> &SynForwardRoute) {
  auto to_core = [&](uint32_t destDelay) -> uint32_t {
    uint32_t global_neuron = (destDelay >> 8) & 0xFFFFFF; // strip delay
    uint32_t core =
        global_neuron > 0
            ? (uint32_t)((global_neuron - 1) / (NEURON_NUM / 8))
            : 0xFFF; // 1-based -> 0-based block, shift by 11 for div by 2048
    return core;
  };

  // Main routing loop over simulation time

router_loop:
  for (int t = 0; t < SimulationTime; t++) {
    bool prev_done = false;
    bool read_axon = true;
    uint32_t coreDone = 0;

    while (!(prev_done)) {
#pragma HLS PIPELINE II = 1 rewind

      // Process main synapse stream
      stream256u_t pkt;
      bool have_pkt = false;
      bool read_axonLoader = false;

      if (SynapseStreamRoute.read_nb(pkt)) {
        have_pkt = true;
        read_axonLoader = false;
      } else if (read_axon) {
        if (SynapseStream.read_nb(pkt)) {
          have_pkt = true;
          read_axonLoader = true;
        }
      }
      if (have_pkt) {
        if (pkt.data[31] == 1) {
          if (read_axonLoader == true) { // read from axonLoader
            SynForwardRoute.write(pkt);
          } else { // read from route stream
            if (coreDone == AmountOfCores - 1) {
              prev_done = true;
            } else {
              coreDone++;
            }

            // Forward if not for this core
            if ((pkt.data.range(29, 8)) != CoreID) {
              SynForwardRoute.write(pkt);
            }
          }
        } else {
          uint32_t dest = to_core(pkt.data.range(29, 0));
          uint32_t dest2 = to_core(pkt.data.range(221, 192));
          if ((dest / 8) != CoreID && dest != 0xFFF) {
            SynForwardRoute.write(pkt);
          }
          if ((dest / 8) == CoreID || (dest2 / 8) == CoreID) {
            uint32_t lane0 = (dest / 8) == CoreID ? (dest % 8) : 0xFFFFFFFF;
            uint32_t lane1 = (dest2 / 8) == CoreID ? (dest2 % 8) : 0xFFFFFFFF;
            if (lane0 == 0 || lane1 == 0)
              SynForwardOut0.write(pkt.data);
            if (lane0 == 1 || lane1 == 1)
              SynForwardOut1.write(pkt.data);
            if (lane0 == 2 || lane1 == 2)
              SynForwardOut2.write(pkt.data);
            if (lane0 == 3 || lane1 == 3)
              SynForwardOut3.write(pkt.data);
            if (lane0 == 4 || lane1 == 4)
              SynForwardOut4.write(pkt.data);
            if (lane0 == 5 || lane1 == 5)
              SynForwardOut5.write(pkt.data);
            if (lane0 == 6 || lane1 == 6)
              SynForwardOut6.write(pkt.data);
            if (lane0 == 7 || lane1 == 7)
              SynForwardOut7.write(pkt.data);
          }
        }
      }
    }

    synapse_list_t pkt_sync;
    pkt_sync = 0;
    SynForwardOut0.write(pkt_sync);
    SynForwardOut1.write(pkt_sync);
    SynForwardOut2.write(pkt_sync);
    SynForwardOut3.write(pkt_sync);
    SynForwardOut4.write(pkt_sync);
    SynForwardOut5.write(pkt_sync);
    SynForwardOut6.write(pkt_sync);
    SynForwardOut7.write(pkt_sync);
  }
}

void Accumulator(hls::stream<synapse_list_t> &SynForward_right,
                 hls::stream<synapse_list_t> &SynForward_left,
                 hls::stream<stream_weight_t> &SpikeOutWeight,
                 uint32_t SimulationTime, uint32_t NeuronStart) {
  //------------------------------------------------------
  //  On‑chip circular buffer to hold delayed packets
  //  Packed: 2 floats (64 bits) per entry to utilize URAM width (72 bits)
  //------------------------------------------------------
  ap_uint<64> buf_flat[(NEURON_NUM / 16) * DELAY];
#pragma HLS bind_storage variable = buf_flat type = ram_2p impl = uram
#pragma HLS DEPENDENCE variable = buf_flat type = inter false

  ap_uint<6> head[NEURON_NUM / 8];
// #pragma HLS bind_storage variable = head type = ram_2p impl = bram

// Initialize weights
// Process 2 neurons per iteration to match packed buffer structure
init_loop:
  for (int i = 0; i < NEURON_NUM / 16; i++) {
    head[2 * i] = 0;
    head[2 * i + 1] = 0;

    for (int j = 0; j < DELAY; j++) {
#pragma HLS PIPELINE II = 1
      buf_flat[i * DELAY + j] = 0;
    }

    // Emit initial 0s for both neurons
    float_to_uint32 temp_conv;
    temp_conv.f = 0.0f;
    stream_weight_t pkt_out;
    pkt_out = temp_conv.u;
    SpikeOutWeight.write(pkt_out); // For neuron 2*i
    SpikeOutWeight.write(pkt_out); // For neuron 2*i+1
  }

// Main time loop - synchronized with SynapseRouter
time_loop:
  for (int t = 0; t < SimulationTime; t++) {
    // Accumulation phase - read synapses until end marker
    bool done_right = false;
    bool done_left = false;

  accumulate_loop:
    while (!done_right || !done_left) {
#pragma HLS PIPELINE II = 8

      ap_uint<512> pkt_new = 0;
      synapse_list_t pkt_right;
      synapse_list_t pkt_left;
      bool have_right = false;
      bool have_left = false;
      if (SynForward_right.read_nb(pkt_right)) {
        pkt_new.range(255, 0) = pkt_right;
        have_right = true;
        if (pkt_right == 0)
          done_right = true;
      }
      if (SynForward_left.read_nb(pkt_left)) {
        pkt_new.range(511, 256) = pkt_left;
        have_left = true;
        if (pkt_left == 0)
          done_left = true;
      }

      if (have_right || have_left) {
        for (int i = 0; i < 8; i++) {
#pragma HLS PIPELINE II = 1
          float_to_uint32 temp_conv;
          Delay_t delay = pkt_new.range(7 + (64 * i), 0 + (64 * i));
          DstID_t dst = pkt_new.range(31 + (64 * i), 8 + (64 * i));
          temp_conv.u = pkt_new.range(63 + (64 * i), 32 + (64 * i));
          if (dst.to_uint() >= NeuronStart &&
              dst.to_uint() < NeuronStart + (NEURON_NUM / 8)) {
            uint32_t idx_buf = (dst.to_uint() - NeuronStart);
            ap_uint<6> h2 = (head[idx_buf] + delay) & 0x3F;

            // Packed access logic
            uint32_t phy_idx = idx_buf >> 1; // idx / 2
            bool upper = idx_buf & 1;        // idx % 2
            uint32_t addr = phy_idx * DELAY + h2;

            ap_uint<64> word = buf_flat[addr];
            uint32_t old_u = upper ? word.range(63, 32) : word.range(31, 0);

            float_to_uint32 conv_u;
            conv_u.u = old_u;
            float new_f = conv_u.f + temp_conv.f;

            conv_u.f = new_f;
            if (upper)
              word.range(63, 32) = conv_u.u;
            else
              word.range(31, 0) = conv_u.u;

            buf_flat[addr] = word;
          }
        }
      }
    }

  // Output phase - write accumulated weights
  // Optimized to read packed word once and output 2 neurons
  output_loop:
    for (int i = 0; i < NEURON_NUM / 16; i++) {
#pragma HLS PIPELINE II = 2

      ap_uint<6> h = head[2 * i];
      uint32_t addr = i * DELAY + h;

      ap_uint<64> word = buf_flat[addr];

      // Neuron 2*i (lower)
      float_to_uint32 conv_lo;
      conv_lo.u = word.range(31, 0);
      stream_weight_t pkt_out1 = conv_lo.u;
      SpikeOutWeight.write(pkt_out1);

      // Neuron 2*i+1 (upper)
      float_to_uint32 conv_hi;
      conv_hi.u = word.range(63, 32);
      stream_weight_t pkt_out2 = conv_hi.u;
      SpikeOutWeight.write(pkt_out2);

      // Clear buffer
      buf_flat[addr] = 0;

      // Advance heads
      head[2 * i] = (head[2 * i] + 1) & 0x3F;
      head[2 * i + 1] = (head[2 * i + 1] + 1) & 0x3F;
    }
  }
}

void weightforward(hls::stream<stream_weight_t> &SpikeWeight0,
                   hls::stream<stream_weight_t> &SpikeWeight1,
                   hls::stream<stream_weight_t> &SpikeWeight2,
                   hls::stream<stream_weight_t> &SpikeWeight3,
                   hls::stream<stream_weight_t> &SpikeWeight4,
                   hls::stream<stream_weight_t> &SpikeWeight5,
                   hls::stream<stream_weight_t> &SpikeWeight6,
                   hls::stream<stream_weight_t> &SpikeWeight7,
                   hls::stream<stream256u_t> &SpikeOutWeight,
                   uint32_t SimulationTime) {
  for (int i = 0; i < NEURON_NUM / 8; i++) {
    stream_weight_t pkt_in[8];
    stream256u_t pkt_out;
    SpikeWeight0.read(pkt_in[0]);
    SpikeWeight1.read(pkt_in[1]);
    SpikeWeight2.read(pkt_in[2]);
    SpikeWeight3.read(pkt_in[3]);
    SpikeWeight4.read(pkt_in[4]);
    SpikeWeight5.read(pkt_in[5]);
    SpikeWeight6.read(pkt_in[6]);
    SpikeWeight7.read(pkt_in[7]);
    pkt_out.data.range(31, 0) = pkt_in[0];
    pkt_out.data.range(63, 32) = pkt_in[1];
    pkt_out.data.range(95, 64) = pkt_in[2];
    pkt_out.data.range(127, 96) = pkt_in[3];
    pkt_out.data.range(159, 128) = pkt_in[4];
    pkt_out.data.range(191, 160) = pkt_in[5];
    pkt_out.data.range(223, 192) = pkt_in[6];
    pkt_out.data.range(255, 224) = pkt_in[7];
    SpikeOutWeight.write(pkt_out);
  }
time_loop:
  for (int t = 0; t < SimulationTime; t++) {
    for (int i = 0; i < NEURON_NUM / 8; i++) {
      stream_weight_t pkt_in[8];
      stream256u_t pkt_out;
      SpikeWeight0.read(pkt_in[0]);
      SpikeWeight1.read(pkt_in[1]);
      SpikeWeight2.read(pkt_in[2]);
      SpikeWeight3.read(pkt_in[3]);
      SpikeWeight4.read(pkt_in[4]);
      SpikeWeight5.read(pkt_in[5]);
      SpikeWeight6.read(pkt_in[6]);
      SpikeWeight7.read(pkt_in[7]);
      pkt_out.data.range(31, 0) = pkt_in[0];
      pkt_out.data.range(63, 32) = pkt_in[1];
      pkt_out.data.range(95, 64) = pkt_in[2];
      pkt_out.data.range(127, 96) = pkt_in[3];
      pkt_out.data.range(159, 128) = pkt_in[4];
      pkt_out.data.range(191, 160) = pkt_in[5];
      pkt_out.data.range(223, 192) = pkt_in[6];
      pkt_out.data.range(255, 224) = pkt_in[7];
      SpikeOutWeight.write(pkt_out);
    }
  }
}

//--------------------------------------------------------------------
//  Top‑level kernel ‒ integrates all sub‑kernels using DATAFLOW
//--------------------------------------------------------------------
extern "C" void SynapseRouter(uint32_t SimulationTime, uint32_t AmountOfCores,
                              uint32_t NeuronStart, uint32_t CoreID,
                              hls::stream<stream256u_t> &syn_route_in_right,
                              hls::stream<stream256u_t> &syn_route_in_left,
                              hls::stream<stream256u_t> &syn_forward_rt_right,
                              hls::stream<stream256u_t> &syn_forward_rt_left,
                              hls::stream<stream256u_t> &synapse_stream_right,
                              hls::stream<stream256u_t> &synapse_stream_left,
                              hls::stream<stream256u_t> &SpikeOutWeight) {
// Interface pragmas
#pragma HLS INTERFACE axis port = synapse_stream_right
#pragma HLS INTERFACE axis port = synapse_stream_left
#pragma HLS INTERFACE axis port = SpikeOutWeight
#pragma HLS INTERFACE axis port = syn_route_in_right
#pragma HLS INTERFACE axis port = syn_route_in_left
#pragma HLS INTERFACE axis port = syn_forward_rt_right
#pragma HLS INTERFACE axis port = syn_forward_rt_left

#pragma HLS INTERFACE s_axilite port = SimulationTime
#pragma HLS INTERFACE s_axilite port = AmountOfCores
#pragma HLS INTERFACE s_axilite port = NeuronStart
#pragma HLS INTERFACE s_axilite port = CoreID
#pragma HLS INTERFACE s_axilite port = return

//---------------------------
//  On‑chip FIFO channels
//  Increased depth for better throughput
//---------------------------
#pragma HLS DATAFLOW

  hls::stream<synapse_list_t> SynForwardOut0("SynForwardOut0");
#pragma HLS STREAM variable = SynForwardOut0 depth = 128
  hls::stream<synapse_list_t> SynForwardOut1("SynForwardOut1");
#pragma HLS STREAM variable = SynForwardOut1 depth = 128
  hls::stream<synapse_list_t> SynForwardOut2("SynForwardOut2");
#pragma HLS STREAM variable = SynForwardOut2 depth = 128
  hls::stream<synapse_list_t> SynForwardOut3("SynForwardOut3");
#pragma HLS STREAM variable = SynForwardOut3 depth = 128
  hls::stream<synapse_list_t> SynForwardOut4("SynForwardOut4");
#pragma HLS STREAM variable = SynForwardOut4 depth = 128
  hls::stream<synapse_list_t> SynForwardOut5("SynForwardOut5");
#pragma HLS STREAM variable = SynForwardOut5 depth = 128
  hls::stream<synapse_list_t> SynForwardOut6("SynForwardOut6");
#pragma HLS STREAM variable = SynForwardOut6 depth = 128
  hls::stream<synapse_list_t> SynForwardOut7("SynForwardOut7");
#pragma HLS STREAM variable = SynForwardOut7 depth = 128

  hls::stream<synapse_list_t> SynForwardOut0_left("SynForwardOut0_left");
#pragma HLS STREAM variable = SynForwardOut0_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut1_left("SynForwardOut1_left");
#pragma HLS STREAM variable = SynForwardOut1_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut2_left("SynForwardOut2_left");
#pragma HLS STREAM variable = SynForwardOut2_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut3_left("SynForwardOut3_left");
#pragma HLS STREAM variable = SynForwardOut3_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut4_left("SynForwardOut4_left");
#pragma HLS STREAM variable = SynForwardOut4_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut5_left("SynForwardOut5_left");
#pragma HLS STREAM variable = SynForwardOut5_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut6_left("SynForwardOut6_left");
#pragma HLS STREAM variable = SynForwardOut6_left depth = 128
  hls::stream<synapse_list_t> SynForwardOut7_left("SynForwardOut7_left");
#pragma HLS STREAM variable = SynForwardOut7_left depth = 128

  hls::stream<stream_weight_t> SpikeWeight0("SpikeWeight0");
#pragma HLS STREAM variable = SpikeWeight0 depth = 32
  hls::stream<stream_weight_t> SpikeWeight1("SpikeWeight1");
#pragma HLS STREAM variable = SpikeWeight1 depth = 32
  hls::stream<stream_weight_t> SpikeWeight2("SpikeWeight2");
#pragma HLS STREAM variable = SpikeWeight2 depth = 32
  hls::stream<stream_weight_t> SpikeWeight3("SpikeWeight3");
#pragma HLS STREAM variable = SpikeWeight3 depth = 32
  hls::stream<stream_weight_t> SpikeWeight4("SpikeWeight4");
#pragma HLS STREAM variable = SpikeWeight4 depth = 32
  hls::stream<stream_weight_t> SpikeWeight5("SpikeWeight5");
#pragma HLS STREAM variable = SpikeWeight5 depth = 32
  hls::stream<stream_weight_t> SpikeWeight6("SpikeWeight6");
#pragma HLS STREAM variable = SpikeWeight6 depth = 32
  hls::stream<stream_weight_t> SpikeWeight7("SpikeWeight7");
#pragma HLS STREAM variable = SpikeWeight7 depth = 32

  RouterRight(synapse_stream_right, syn_route_in_right, SimulationTime,
              AmountOfCores, CoreID, SynForwardOut0, SynForwardOut1,
              SynForwardOut2, SynForwardOut3, SynForwardOut4, SynForwardOut5,
              SynForwardOut6, SynForwardOut7, syn_forward_rt_right);

  RouterLeft(synapse_stream_left, syn_route_in_left, SimulationTime,
             AmountOfCores, CoreID, SynForwardOut0_left, SynForwardOut1_left,
             SynForwardOut2_left, SynForwardOut3_left, SynForwardOut4_left,
             SynForwardOut5_left, SynForwardOut6_left, SynForwardOut7_left,
             syn_forward_rt_left);

Accumulator0:
  Accumulator(SynForwardOut0, SynForwardOut0_left, SpikeWeight0, SimulationTime,
              NeuronStart);
Accumulator1:
  Accumulator(SynForwardOut1, SynForwardOut1_left, SpikeWeight1, SimulationTime,
              NeuronStart + (NEURON_NUM / 8));
Accumulator2:
  Accumulator(SynForwardOut2, SynForwardOut2_left, SpikeWeight2, SimulationTime,
              NeuronStart + ((NEURON_NUM / 8) * 2));
Accumulator3:
  Accumulator(SynForwardOut3, SynForwardOut3_left, SpikeWeight3, SimulationTime,
              NeuronStart + ((NEURON_NUM / 8) * 3));
Accumulator4:
  Accumulator(SynForwardOut4, SynForwardOut4_left, SpikeWeight4, SimulationTime,
              NeuronStart + ((NEURON_NUM / 8) * 4));
Accumulator5:
  Accumulator(SynForwardOut5, SynForwardOut5_left, SpikeWeight5, SimulationTime,
              NeuronStart + ((NEURON_NUM / 8) * 5));
Accumulator6:
  Accumulator(SynForwardOut6, SynForwardOut6_left, SpikeWeight6, SimulationTime,
              NeuronStart + ((NEURON_NUM / 8) * 6));
Accumulator7:
  Accumulator(SynForwardOut7, SynForwardOut7_left, SpikeWeight7, SimulationTime,
              NeuronStart + ((NEURON_NUM / 8) * 7));

  weightforward(SpikeWeight0, SpikeWeight1, SpikeWeight2, SpikeWeight3,
                SpikeWeight4, SpikeWeight5, SpikeWeight6, SpikeWeight7,
                SpikeOutWeight, SimulationTime);
}

//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================
