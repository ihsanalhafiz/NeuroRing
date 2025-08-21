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

#ifndef SW_SIM
#define _XF_SYNTHESIS_ 1
#endif

#define BUF_IDX(core, ofs)   ((core)*DELAY + (ofs))

#define NEURON_NUM_X 2048
#define SYNAPSE_LIST_SIZE_X 100

//--------------------------------------------------------------------
//  Top‑level kernel ‒ integrates all sub‑kernels using DATAFLOW
//--------------------------------------------------------------------
extern "C" void NeuroRing_singlestep(
    uint32_t                    *SynapseList,
    uint32_t                    *SpikeRecorder,
    uint32_t                     SimulationTime,
    float                        threshold,
    float                        membrane_potential,
    uint32_t                     AmountOfCores,
    uint32_t                     NeuronStart,
    uint32_t                     NeuronTotal,
    uint32_t                     DCstimStart,
    uint32_t                     DCstimTotal,
    float                        DCstimAmp,
    uint32_t                     CoreID)
{
    

    // define stream variables
    hls::stream<stream512u_t> SynapseStream;
    hls::stream<stream512u_t> SynapseStreamRoute;
    hls::stream<stream512u_t> SynForwardRoute;
    hls::stream<synapse_word_t> SynForward;
    hls::stream<synapse_word_t> SynForward1;
    hls::stream<synapse_word_t> SynForward2;
    hls::stream<synapse_word_t> SynForward3;
    hls::stream<synapse_word_t> SynForward4;
    hls::stream<synapse_word_t> SynForward5;
    hls::stream<synapse_word_t> SynForward6;
    hls::stream<synapse_word_t> SynForward7;

    // Helper function to create synapse packet from vector
    auto create_synapse_packet = [](const hls::vector<uint32_t, 16>& data) -> stream512u_t {
        stream512u_t packet = {};
        packet.data = 0;
        for (int k = 0; k < 16; k++) {
            packet.data.range(511 - k*32, 480 - k*32) = data[k];
        }
        return packet;
    };

    uint32_t DCstim_float[NEURON_NUM];
    uint32_t SynapseSize[NEURON_NUM];
    uint32_t UmemPot[NEURON_NUM];

    // Helper function to write packet to stream
    auto write_packet_to_stream = [](hls::stream<stream512u_t>& stream, const stream512u_t& packet) {
        bool write_status = false;
        while (!write_status) {
            write_status = stream.write_nb(packet);
        }
    };

    // read parameters from file
    for (int i = 0; i < NeuronTotal; i++) {
        // read 16 data from SpikeRecorder_SynapseList
        hls::vector<uint32_t, 16> parameter_data;
        #pragma HLS ARRAY_PARTITION variable=parameter_data complete dim=1
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL
            parameter_data[k] = SynapseList[i*SYNAPSE_LIST_SIZE + k];
        }
        SynapseSize[i] = parameter_data[0]*2;
        DCstim_float[i] = parameter_data[1];
        UmemPot[i] = parameter_data[2];
    }

    // send data of UmemPot to SynapseStream (8 lanes per 512-bit packet)
    for (int i = 0; i < NeuronTotal; i+=8) {
        // UmemPot as weight, 0xFC as delay, and index as destination
        stream512u_t packet;
        packet.data = 0;
        for (int j = 0; j < 8; j++) {
            int base_bit = 511 - j * 64;
            bool valid_neuron = (i+j < NeuronTotal);
            packet.data.range(base_bit, base_bit - 23) = valid_neuron ? i+j+NeuronStart : 0;
            packet.data.range(base_bit - 24, base_bit - 31) = 0xFC;
            packet.data.range(base_bit - 32, base_bit - 63) = valid_neuron ? UmemPot[i+j] : 0;
        }
        write_packet_to_stream(SynapseStream, packet);
    }

    // Initialize SynapseRouter ------------------------------------------------------------
    
    ap_uint<24> start[8];
    ap_uint<24> end[8];
    #pragma HLS ARRAY_PARTITION variable=start complete
    #pragma HLS ARRAY_PARTITION variable=end complete

    start[0] = 1 + 2048*CoreID;
    start[1] = 257 + 2048*(CoreID);
    start[2] = 513 + 2048*(CoreID);
    start[3] = 769 + 2048*(CoreID);
    start[4] = 1025 + 2048*(CoreID);
    start[5] = 1281 + 2048*(CoreID);
    start[6] = 1537 + 2048*(CoreID);
    start[7] = 1793 + 2048*(CoreID);

    end[0] = 257 + 2048*(CoreID); 
    end[1] = 513 + 2048*(CoreID);
    end[2] = 769 + 2048*(CoreID);
    end[3] = 1025 + 2048*(CoreID);
    end[4] = 1281 + 2048*(CoreID);
    end[5] = 1537 + 2048*(CoreID);
    end[6] = 1793 + 2048*(CoreID);
    end[7] = 2049 + 2048*(CoreID);

    // Initialize SomaEngine ------------------------------------------------------------
    const float dt = 0.1f;
    const float tau_m = 10.0f;
    const float tau_syn = 0.5f;
    const float C_m = 250.0f;
    const float E_L = -65.0f;
    const float V_decay = 0.99004983f;   // exp(-dt/tau_m)
    const float I_decay = 0.81873075f;   // exp(-dt/tau_syn)
    const float syn_to_vm = (1.0f/C_m) * ((I_decay - V_decay) / ((1.0f/tau_m) - (1.0f/tau_syn)));
    const int   t_ref_steps = 20;        // round(2.0/0.1)
    float threshold_float = threshold;
    float membrane_potential_float = membrane_potential;

    // SomaEngine0 ------------------------------------------------------------

    float buf_flat[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat type=ram_2p impl=uram
    ap_uint<6> head[NCORE];
    bool runstate = true;
    float U_membPot[NCORE];   // membrane potential (mV)
    float I_PreSynCurr[NCORE];
    uint16_t R_RefCnt[NCORE];
    float x_state[NCORE];
    float C_acc[NCORE];
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

    // SomaEngine1 ------------------------------------------------------------

    float buf_flat1[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat1 type=ram_2p impl=uram
    ap_uint<6> head1[NCORE];
    bool runstate1 = true;
    float U_membPot1[NCORE];   // membrane potential (mV)
    float I_PreSynCurr1[NCORE];
    uint16_t R_RefCnt1[NCORE];
    float x_state1[NCORE];
    float C_acc1[NCORE];
    init_loop_outer1: for (int core = 0; core < NCORE; core++) {
        U_membPot1[core] = membrane_potential_float;
        I_PreSynCurr1[core] = 0;
        R_RefCnt1[core] = 0;
        x_state1[core] = 0;
        C_acc1[core] = 0;
        head1[core] = 0;
        init_loop_inner1: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat1[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // SomaEngine2 ------------------------------------------------------------

    float buf_flat2[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat2 type=ram_2p impl=uram
    ap_uint<6> head2[NCORE];
    bool runstate2 = true;
    float U_membPot2[NCORE];   // membrane potential (mV)
    float I_PreSynCurr2[NCORE];
    uint16_t R_RefCnt2[NCORE];
    float x_state2[NCORE];
    float C_acc2[NCORE];
    init_loop_outer2: for (int core = 0; core < NCORE; core++) {
        U_membPot2[core] = membrane_potential_float;
        I_PreSynCurr2[core] = 0;
        R_RefCnt2[core] = 0;
        x_state2[core] = 0;
        C_acc2[core] = 0;
        head2[core] = 0;
        init_loop_inner2: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat2[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // SomaEngine3 ------------------------------------------------------------

    float buf_flat3[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat3 type=ram_2p impl=uram
    ap_uint<6> head3[NCORE];
    bool runstate3 = true;
    float U_membPot3[NCORE];   // membrane potential (mV)
    float I_PreSynCurr3[NCORE];
    uint16_t R_RefCnt3[NCORE];
    float x_state3[NCORE];
    float C_acc3[NCORE];
    init_loop_outer3: for (int core = 0; core < NCORE; core++) {
        U_membPot3[core] = membrane_potential_float;
        I_PreSynCurr3[core] = 0;
        R_RefCnt3[core] = 0;
        x_state3[core] = 0;
        C_acc3[core] = 0;
        head3[core] = 0;
        init_loop_inner3: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat3[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // SomaEngine4 ------------------------------------------------------------

    float buf_flat4[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat4 type=ram_2p impl=uram
    ap_uint<6> head4[NCORE];
    bool runstate4 = true;
    float U_membPot4[NCORE];   // membrane potential (mV)
    float I_PreSynCurr4[NCORE];
    uint16_t R_RefCnt4[NCORE];
    float x_state4[NCORE];
    float C_acc4[NCORE];
    init_loop_outer4: for (int core = 0; core < NCORE; core++) {
        U_membPot4[core] = membrane_potential_float;
        I_PreSynCurr4[core] = 0;
        R_RefCnt4[core] = 0;
        x_state4[core] = 0;
        C_acc4[core] = 0;
        head4[core] = 0;
        init_loop_inner4: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat4[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // SomaEngine5 ------------------------------------------------------------

    float buf_flat5[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat5 type=ram_2p impl=uram
    ap_uint<6> head5[NCORE];
    bool runstate5 = true;
    float U_membPot5[NCORE];   // membrane potential (mV)
    float I_PreSynCurr5[NCORE];
    uint16_t R_RefCnt5[NCORE];
    float x_state5[NCORE];
    float C_acc5[NCORE];
    init_loop_outer5: for (int core = 0; core < NCORE; core++) {
        U_membPot5[core] = membrane_potential_float;
        I_PreSynCurr5[core] = 0;
        R_RefCnt5[core] = 0;
        x_state5[core] = 0;
        C_acc5[core] = 0;
        head5[core] = 0;
        init_loop_inner5: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat5[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // SomaEngine6 ------------------------------------------------------------

    float buf_flat6[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat6 type=ram_2p impl=uram
    ap_uint<6> head6[NCORE];
    bool runstate6 = true;
    float U_membPot6[NCORE];   // membrane potential (mV)
    float I_PreSynCurr6[NCORE];
    uint16_t R_RefCnt6[NCORE];
    float x_state6[NCORE];
    float C_acc6[NCORE];
    init_loop_outer6: for (int core = 0; core < NCORE; core++) {
        U_membPot6[core] = membrane_potential_float;
        I_PreSynCurr6[core] = 0;
        R_RefCnt6[core] = 0;
        x_state6[core] = 0;
        C_acc6[core] = 0;
        head6[core] = 0;
        init_loop_inner6: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat6[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // SomaEngine7 ------------------------------------------------------------

    float buf_flat7[NCORE*DELAY];
    #pragma HLS bind_storage variable=buf_flat7 type=ram_2p impl=uram
    ap_uint<6> head7[NCORE];
    bool runstate7 = true;
    float U_membPot7[NCORE];   // membrane potential (mV)
    float I_PreSynCurr7[NCORE];
    uint16_t R_RefCnt7[NCORE];
    float x_state7[NCORE];
    float C_acc7[NCORE];
    init_loop_outer7: for (int core = 0; core < NCORE; core++) {
        U_membPot7[core] = membrane_potential_float;
        I_PreSynCurr7[core] = 0;
        R_RefCnt7[core] = 0;
        x_state7[core] = 0;
        C_acc7[core] = 0;
        head7[core] = 0;
        init_loop_inner7: for (int d = 0; d < DELAY; d++) {
            #pragma HLS UNROLL factor=NLANE
            buf_flat7[BUF_IDX(core,d)] = 0.0f;
        }
    }

    // print total neuron
    std::cout << "Total neuron: " << NeuronTotal << std::endl;

    ////////////////////----------------------------------------------------------////////////////////
    // Main loop
    ////////////////////----------------------------------------------------------////////////////////

    Main_loop: for (int t = 0; t < SimulationTime; t++) {
        // read spike (assemble full 2048-bit bitmap from 64x32b words)
        stream2048u_t spike_read;
        for (int i = 0; i < 64; i++) {
            spike_read.data.range(((i+1)*32 - 1), (i*32)) = SpikeRecorder[t*64 + i];
        }

        // process each neuron that fired
        for (int i = 0; i < NeuronTotal; i++) {
            if (spike_read.data.range(i, i) == 1) {
                // process synapses in chunks of 16
                for (int j = 1; j < (SynapseSize[i] + 15) / 16; j++) {
                    // Read 16 synapses at once
                    hls::vector<uint32_t, 16> synapse_data;
                    #pragma HLS ARRAY_PARTITION variable=synapse_data complete dim=1
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS UNROLL
                        synapse_data[k] = SynapseList[i*SYNAPSE_LIST_SIZE + j*16 + k];
                    }
                    // Create and send packet
                    stream512u_t packet = create_synapse_packet(synapse_data);
                    write_packet_to_stream(SynapseStream, packet);
                }
            }
        }

        if (t >= DCstimStart && t < DCstimStart + DCstimTotal) {
            for (int i = NeuronStart; i < (int)(NeuronStart + NeuronTotal); i += 8) {
                // Use DCstimAmp uniformly; can be replaced with DCstim_float[i-NeuronStart] if per-neuron values are desired
                stream512u_t packet;
                packet.data = 0;
                for (int j = 0; j < 8; j++) {
                    int base_bit = 511 - j * 64;
                    bool valid_neuron = (i+j < (NeuronStart + NeuronTotal));
                    packet.data.range(base_bit, base_bit - 23) = valid_neuron ? i+j : 0;
                    packet.data.range(base_bit - 24, base_bit - 31) = 0x00;
                    packet.data.range(base_bit - 32, base_bit - 63) = valid_neuron ? DCstim_float[(i+j)-NeuronStart] : 0;
                }
                write_packet_to_stream(SynapseStream, packet);
            }
        }

        // Send sync word in lane 0: dst=NeuronStart, delay=0xFE, weight=0
        stream512u_t sync_packet;
        sync_packet.data = 0;
        uint32_t dst_delay_sync = (((NeuronStart) << 8) & 0xFFFFFF00) | 0xFE;
        sync_packet.data.range(511, 480) = dst_delay_sync;
        write_packet_to_stream(SynapseStream, sync_packet);

        ////////////////////----------------------------------------------------------////////////////////
        // end AxonLoader
        ////////////////////----------------------------------------------------------////////////////////

        ////////////////////----------------------------------------------------------////////////////////
        // start SynapseRouter
        ////////////////////----------------------------------------------------------////////////////////
        bool axon_done = false;
        bool prev_done = false;
        uint32_t coreDone = 0;
        
        while (!(axon_done && prev_done)) {
        #pragma HLS PIPELINE II=1 rewind
            
            // Helper to write to synapse streams (block until success)
            auto write_synapse_to_stream = [](hls::stream<synapse_word_t>& stream, const synapse_word_t& packet) {
                bool write_status = false;
                while (!write_status) {
                    write_status = stream.write_nb(packet);
                }
            };

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
                
                // Unpack all 16 synapses at once
                for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                    int base_bit = 511 - i * 64;
                    dst[i] = pkt.data.range(base_bit, base_bit - 23);
                    delay[i] = pkt.data.range(base_bit - 24, base_bit - 31);
                    weight_bits[i] = pkt.data.range(base_bit - 32, base_bit - 63);
                }

                // Check if this is an axon done signal
                if (delay[0] == 0xFE) {
                    if((dst[0] == NeuronStart) && axon_done == false) {
                        axon_done = true;
                        prev_done = true; // single-core sim: consider previous done as soon as self done
                    } else{
                        // Ignore non-local done in single-step
                        prev_done = true;
                    }
                }else{ // end delay[0] == 0xFE
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

                        // Dispatch to the right stream (prefer blocking write to avoid spin-wait)
                        switch (region) {
                            case 0: write_synapse_to_stream(SynForward, temp); break;
                            case 1: write_synapse_to_stream(SynForward1, temp); break;
                            case 2: write_synapse_to_stream(SynForward2, temp); break;
                            case 3: write_synapse_to_stream(SynForward3, temp); break;
                            case 4: write_synapse_to_stream(SynForward4, temp); break;
                            case 5: write_synapse_to_stream(SynForward5, temp); break;
                            case 6: write_synapse_to_stream(SynForward6, temp); break;
                            case 7: write_synapse_to_stream(SynForward7, temp); break;
                            default: break; // no-op if no region matched
                        }
                        if(region != -1) {
                            // Clear once
                            dst[i] = 0;
                            delay[i] = 0;
                            weight_bits[i] = 0;
                        }
                    }
                    bool any_non_zero = false;
                    for(int i = 0; i < 8; i++) {
                        any_non_zero = any_non_zero || (dst[i] != 0);
                    }
                    if(any_non_zero) {
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
            } // end if have_pkt
        } // end while loop
        
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

        std::cout << "Size of SynForward: " << SynForward.size() << std::endl;
        std::cout << "Size of SynForward1: " << SynForward1.size() << std::endl;
        std::cout << "Size of SynForward2: " << SynForward2.size() << std::endl;
        std::cout << "Size of SynForward3: " << SynForward3.size() << std::endl;
        std::cout << "Size of SynForward4: " << SynForward4.size() << std::endl;
        std::cout << "Size of SynForward5: " << SynForward5.size() << std::endl;
        std::cout << "Size of SynForward6: " << SynForward6.size() << std::endl;
        std::cout << "Size of SynForward7: " << SynForward7.size() << std::endl;

        ////////////////////----------------------------------------------------------////////////////////
        // end SynapseRouter
        ////////////////////----------------------------------------------------------////////////////////

        ////////////////////----------------------------------------------------------////////////////////
        // start SomaEngine
        ////////////////////----------------------------------------------------------////////////////////
        // SomaEngine 0 ------------------------------------------------------------
        bool done = false;
        while (!done) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot[dst.to_uint()-start[0]] = weight;
                }
                else if (delay == 0xFE) {
                    done = true;
                } else {
                    ap_uint<6> h2 = (head[dst.to_int()-start[0]] + delay) & 0x3F;
                    float weight2 = buf_flat[BUF_IDX((dst.to_int()-start[0]),h2)];    
                    buf_flat[BUF_IDX((dst.to_int()-start[0]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head[i]) & 0x3F;
            I_PreSynCurr[i] = I_PreSynCurr[i] * I_decay + buf_flat[BUF_IDX(i,h3)];
            buf_flat[BUF_IDX(i,h3)] = 0;
            head[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status = 0;

        for(int i = 0; i < NCORE; i++) {
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
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*0 + i] = spike_status.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 1 ------------------------------------------------------------

        bool done1 = false;
        while (!done1) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward1.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot1[dst.to_uint()-start[1]] = weight;
                }
                else if (delay == 0xFE) {
                    done1 = true;
                } else {
                    ap_uint<6> h2 = (head1[dst.to_int()-start[1]] + delay) & 0x3F;
                    float weight2 = buf_flat1[BUF_IDX((dst.to_int()-start[1]),h2)];    
                    buf_flat1[BUF_IDX((dst.to_int()-start[1]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head1[i]) & 0x3F;
            I_PreSynCurr1[i] = I_PreSynCurr1[i] * I_decay + buf_flat1[BUF_IDX(i,h3)];
            buf_flat1[BUF_IDX(i,h3)] = 0;
            head1[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status1 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt1[i] > 0) {
                R_RefCnt1[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot1[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot1[i];
                float i_prev = I_PreSynCurr1[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot1[i] = v_new;
                // Spike generation
                if (U_membPot1[i] >= threshold_float) {
                    spike_status1.range(i, i) = 1;
                    U_membPot1[i] = membrane_potential_float;
                    R_RefCnt1[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*1 + i] = spike_status1.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 2 ------------------------------------------------------------
        
        bool done2 = false;
        while (!done2) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward2.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot2[dst.to_uint()-start[2]] = weight;
                }
                else if (delay == 0xFE) {
                    done2 = true;
                } else {
                    ap_uint<6> h2 = (head2[dst.to_int()-start[2]] + delay) & 0x3F;
                    float weight2 = buf_flat2[BUF_IDX((dst.to_int()-start[2]),h2)];    
                    buf_flat2[BUF_IDX((dst.to_int()-start[2]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head2[i]) & 0x3F;
            I_PreSynCurr2[i] = I_PreSynCurr2[i] * I_decay + buf_flat2[BUF_IDX(i,h3)];
            buf_flat2[BUF_IDX(i,h3)] = 0;
            head2[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status2 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt2[i] > 0) {
                R_RefCnt2[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot2[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot2[i];
                float i_prev = I_PreSynCurr2[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot2[i] = v_new;
                // Spike generation
                if (U_membPot2[i] >= threshold_float) {
                    spike_status2.range(i, i) = 1;
                    U_membPot2[i] = membrane_potential_float;
                    R_RefCnt2[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*2 + i] = spike_status2.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 3 ------------------------------------------------------------

        bool done3 = false;
        while (!done3) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward3.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot3[dst.to_uint()-start[3]] = weight;
                }
                else if (delay == 0xFE) {
                    done3 = true;
                } else {
                    ap_uint<6> h2 = (head3[dst.to_int()-start[3]] + delay) & 0x3F;
                    float weight2 = buf_flat3[BUF_IDX((dst.to_int()-start[3]),h2)];    
                    buf_flat3[BUF_IDX((dst.to_int()-start[3]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head3[i]) & 0x3F;
            I_PreSynCurr3[i] = I_PreSynCurr3[i] * I_decay + buf_flat3[BUF_IDX(i,h3)];
            buf_flat3[BUF_IDX(i,h3)] = 0;
            head3[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status3 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt3[i] > 0) {
                R_RefCnt3[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot3[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot3[i];
                float i_prev = I_PreSynCurr3[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot3[i] = v_new;
                // Spike generation
                if (U_membPot3[i] >= threshold_float) {
                    spike_status3.range(i, i) = 1;
                    U_membPot3[i] = membrane_potential_float;
                    R_RefCnt3[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*3 + i] = spike_status3.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 4 ------------------------------------------------------------
        
        bool done4 = false;
        while (!done4) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward4.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot4[dst.to_uint()-start[4]] = weight;
                }
                else if (delay == 0xFE) {
                    done4 = true;
                } else {
                    ap_uint<6> h2 = (head4[dst.to_int()-start[4]] + delay) & 0x3F;
                    float weight2 = buf_flat4[BUF_IDX((dst.to_int()-start[4]),h2)];    
                    buf_flat4[BUF_IDX((dst.to_int()-start[4]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head4[i]) & 0x3F;
            I_PreSynCurr4[i] = I_PreSynCurr4[i] * I_decay + buf_flat4[BUF_IDX(i,h3)];
            buf_flat4[BUF_IDX(i,h3)] = 0;
            head4[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status4 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt4[i] > 0) {
                R_RefCnt4[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot4[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot4[i];
                float i_prev = I_PreSynCurr4[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot4[i] = v_new;
                // Spike generation
                if (U_membPot4[i] >= threshold_float) {
                    spike_status4.range(i, i) = 1;
                    U_membPot4[i] = membrane_potential_float;
                    R_RefCnt4[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*4 + i] = spike_status4.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 5 ------------------------------------------------------------
        
        bool done5 = false;
        while (!done5) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward5.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot5[dst.to_uint()-start[5]] = weight;
                }
                else if (delay == 0xFE) {
                    done5 = true;
                } else {
                    ap_uint<6> h2 = (head5[dst.to_int()-start[5]] + delay) & 0x3F;
                    float weight2 = buf_flat5[BUF_IDX((dst.to_int()-start[5]),h2)];    
                    buf_flat5[BUF_IDX((dst.to_int()-start[5]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head5[i]) & 0x3F;
            I_PreSynCurr5[i] = I_PreSynCurr5[i] * I_decay + buf_flat5[BUF_IDX(i,h3)];
            buf_flat5[BUF_IDX(i,h3)] = 0;
            head5[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status5 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt5[i] > 0) {
                R_RefCnt5[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot5[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot5[i];
                float i_prev = I_PreSynCurr5[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot5[i] = v_new;
                // Spike generation
                if (U_membPot5[i] >= threshold_float) {
                    spike_status5.range(i, i) = 1;
                    U_membPot5[i] = membrane_potential_float;
                    R_RefCnt5[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*5 + i] = spike_status5.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 6 ------------------------------------------------------------
        
        bool done6 = false;
        while (!done6) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward6.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot6[dst.to_uint()-start[6]] = weight;
                }
                else if (delay == 0xFE) {
                    done6 = true;
                } else {
                    ap_uint<6> h2 = (head6[dst.to_int()-start[6]] + delay) & 0x3F;
                    float weight2 = buf_flat6[BUF_IDX((dst.to_int()-start[6]),h2)];    
                    buf_flat6[BUF_IDX((dst.to_int()-start[6]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head6[i]) & 0x3F;
            I_PreSynCurr6[i] = I_PreSynCurr6[i] * I_decay + buf_flat6[BUF_IDX(i,h3)];
            buf_flat6[BUF_IDX(i,h3)] = 0;
            head6[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status6 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt6[i] > 0) {
                R_RefCnt6[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot6[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot6[i];
                float i_prev = I_PreSynCurr6[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot6[i] = v_new;
                // Spike generation
                if (U_membPot6[i] >= threshold_float) {
                    spike_status6.range(i, i) = 1;
                    U_membPot6[i] = membrane_potential_float;
                    R_RefCnt6[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*6 + i] = spike_status6.range(((i+1)*32 - 1), (i*32));
        }

        // SomaEngine 7 ------------------------------------------------------------
        
        bool done7 = false;
        while (!done7) {
            //#pragma HLS PIPELINE II=6 rewind
            //--------------------------------------------------
            // 2) Accept new packets from SpikeStream
            //--------------------------------------------------
            synapse_word_t pkt_new;
            bool have_pkt = SynForward7.read_nb(pkt_new);
            if (have_pkt) {
                DstID_t dst = pkt_new.range(63, 40);
                Delay_t delay = pkt_new.range(39, 32);
                float_to_uint32 temp_conv;
                temp_conv.u = pkt_new.range(31, 0);
                float weight = temp_conv.f;
                if (delay == 0xFC) {
                    U_membPot7[dst.to_uint()-start[7]] = weight;
                }
                else if (delay == 0xFE) {
                    done7 = true;
                } else {
                    ap_uint<6> h2 = (head7[dst.to_int()-start[7]] + delay) & 0x3F;
                    float weight2 = buf_flat7[BUF_IDX((dst.to_int()-start[7]),h2)];    
                    buf_flat7[BUF_IDX((dst.to_int()-start[7]),h2)] = weight2 + weight;
                }
            }
        }
        //--------------------------------------------------
        // 3) Update spike_status[] based on neuron PE results
        //--------------------------------------------------
        for(int i = 0; i < NCORE; i++) {
            // Exponential synaptic current decay and accumulation of inputs (pA)
            ap_uint<6> h3 = (head7[i]) & 0x3F;
            I_PreSynCurr7[i] = I_PreSynCurr7[i] * I_decay + buf_flat7[BUF_IDX(i,h3)];
            buf_flat7[BUF_IDX(i,h3)] = 0;
            head7[i] = (h3 + 1) & 0x3F;
        }

        ap_uint<256> spike_status7 = 0;

        for(int i = 0; i < NCORE; i++) {
            // Refractory countdown
            if (R_RefCnt7[i] > 0) {
                R_RefCnt7[i] -= 1;
                // Clamp to V_reset during refractory
                U_membPot7[i] = membrane_potential_float;
            } else {
                float v_prev = U_membPot7[i];
                float i_prev = I_PreSynCurr7[i]; // pA
                float v_new = E_L + (v_prev - E_L) * V_decay + i_prev * syn_to_vm;
                U_membPot7[i] = v_new;
                // Spike generation
                if (U_membPot7[i] >= threshold_float) {
                    spike_status7.range(i, i) = 1;
                    U_membPot7[i] = membrane_potential_float;
                    R_RefCnt7[i] = (uint16_t)t_ref_steps;
                }
            }
        }
        //write spike_status to SpikeRecorder
        for(int i = 0; i < 8; i++) {
            SpikeRecorder[(t+1)*64 + 8*7 + i] = spike_status7.range(((i+1)*32 - 1), (i*32));
        }

        //print t
        std::cout << "t: " << t << std::endl;
        // print size of SynForward1, SynForward2, SynForward3, SynForward4, SynForward5, SynForward6, SynForward7
        std::cout << "Size of SynForward: " << SynForward.size() << std::endl;
        std::cout << "Size of SynForward1: " << SynForward1.size() << std::endl;
        std::cout << "Size of SynForward2: " << SynForward2.size() << std::endl;
        std::cout << "Size of SynForward3: " << SynForward3.size() << std::endl;
        std::cout << "Size of SynForward4: " << SynForward4.size() << std::endl;
        std::cout << "Size of SynForward5: " << SynForward5.size() << std::endl;
        std::cout << "Size of SynForward6: " << SynForward6.size() << std::endl;
        std::cout << "Size of SynForward7: " << SynForward7.size() << std::endl;
        std::cout << "--------------------------------" << std::endl;
        
    } // end of loop t < SimulationTime
} // end of NeuroRing_singlestep



//============================================================
//  END OF FILE – fill out TODOs & tune pragmas for your design
//============================================================

// Main function for testing NeuroRing_singlestep
#ifdef _XF_SYNTHESIS_
// Skip main function during HLS synthesis
#else
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int main() {
    std::cout << "Starting NeuroRing_singlestep simulation..." << std::endl;
    
    // Simulation parameters
    const uint32_t SimulationTime = 5;
    const float threshold = -50.0f;
    const float membrane_potential = -65.0f;
    const uint32_t AmountOfCores = 1;
    const uint32_t NeuronStart = 1;
    const uint32_t NeuronTotal = 771;
    const uint32_t DCstimStart = 0;
    const uint32_t DCstimTotal = 50;
    const float DCstimAmp = 170.0f;
    
    // Calculate array sizes
    const uint32_t SynapseListSize = NeuronTotal * SYNAPSE_LIST_SIZE;  // 771 * 10000
    const uint32_t SpikeRecorderSize = 64 * SimulationTime;  // 64 * 1000
    
    std::cout << "Allocating SynapseList array of size: " << SynapseListSize << std::endl;
    std::cout << "Allocating SpikeRecorder array of size: " << SpikeRecorderSize << std::endl;
    
    // Allocate and initialize arrays
    uint32_t* SynapseList = new uint32_t[SynapseListSize];
    uint32_t* SpikeRecorder = new uint32_t[SpikeRecorderSize];
    
    // Initialize SpikeRecorder to 0
    for (uint32_t i = 0; i < SpikeRecorderSize; i++) {
        SpikeRecorder[i] = 0;
    }
    
    // Read Synapse_list.csv
    std::cout << "Reading Synapse_list.csv..." << std::endl;
    std::ifstream csvFile("host_py/Synapse_list.csv");
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open Synapse_list.csv" << std::endl;
        delete[] SynapseList;
        delete[] SpikeRecorder;
        return -1;
    }
    
    std::string line;
    uint32_t lineCount = 0;
    uint32_t arrayIndex = 0;
    
    while (std::getline(csvFile, line) && arrayIndex < SynapseListSize) {
        if (!line.empty()) {
            try {
                SynapseList[arrayIndex] = std::stoul(line);
                arrayIndex++;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line " << lineCount << ": " << line << std::endl;
            }
        }
        lineCount++;
    }
    
    csvFile.close();
    std::cout << "Read " << arrayIndex << " values from CSV (expected: " << SynapseListSize << ")" << std::endl;
    
    if (arrayIndex != SynapseListSize) {
        std::cerr << "Warning: Expected " << SynapseListSize << " values but got " << arrayIndex << std::endl;
    }
    
    // Call the simulation function
    std::cout << "Starting simulation..." << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  SimulationTime: " << SimulationTime << std::endl;
    std::cout << "  threshold: " << threshold << std::endl;
    std::cout << "  membrane_potential: " << membrane_potential << std::endl;
    std::cout << "  AmountOfCores: " << AmountOfCores << std::endl;
    std::cout << "  NeuronStart: " << NeuronStart << std::endl;
    std::cout << "  NeuronTotal: " << NeuronTotal << std::endl;
    std::cout << "  DCstimStart: " << DCstimStart << std::endl;
    std::cout << "  DCstimTotal: " << DCstimTotal << std::endl;
    std::cout << "  DCstimAmp: " << DCstimAmp << std::endl;
    
    NeuroRing_singlestep(
        SynapseList,
        SpikeRecorder,
        SimulationTime,
        threshold,
        membrane_potential,
        AmountOfCores,
        NeuronStart,
        NeuronTotal,
        DCstimStart,
        DCstimTotal,
        DCstimAmp,
        0
    );
    
    std::cout << "Simulation completed!" << std::endl;
    
    // Print some sample results from SpikeRecorder
    std::cout << "\nSample SpikeRecorder results:" << std::endl;
    for (uint32_t t = 0; t < SimulationTime; t++) {  // Show first 5 timesteps
        std::cout << "Timestep " << t << ": ";
        for (uint32_t i = 0; i < 8 ; i++) {  // Show first 8 words
            std::cout << "Core " << i << ": ";
            for (uint32_t j = 0; j < 8; j++) {
                std::cout << SpikeRecorder[t * 64 + i * 8 + j] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    delete[] SynapseList;
    delete[] SpikeRecorder;
    
    std::cout << "Simulation finished successfully!" << std::endl;
    return 0;
}
#endif
