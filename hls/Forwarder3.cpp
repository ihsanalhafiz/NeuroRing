#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_vector.h>
#include <ap_axi_sdata.h>
#include "NeuroRing.h"

#define _XF_SYNTHESIS_ 1

//====================================================================
//  SynapseRouter – Routes synapse data to 4 slots based on destination
//====================================================================
extern "C" void Forwarder3(
    hls::stream<stream512u_t>    &SynapseStreamIn,
    hls::stream<stream512u_t>    &SynapseRouteIn,
    hls::stream<stream512u_t>    &SynapseStreamOut,
    hls::stream<stream512u_t>    &SynapseRouteOut)
{
    #pragma HLS INTERFACE axis port=SynapseStreamIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseRouteIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseStreamOut bundle=AXIS_OUT
    #pragma HLS INTERFACE axis port=SynapseRouteOut bundle=AXIS_OUT
    #pragma HLS INTERFACE ap_ctrl_none port=return

    const ap_uint<8> id = 3;

    bool last_packet[4] = {false, false, false, false};
    bool while_loop = true;
    stream512u_t pkt;

    while(while_loop) {
        bool read_synapseStreamIn = false;
        read_synapseStreamIn = SynapseStreamIn.read_nb(pkt);
        if(read_synapseStreamIn) {
            if(pkt.dest == id) {
                last_packet[pkt.id] = pkt.last;
                SynapseStreamOut.write(pkt);
            }
            else {
                SynapseRouteOut.write(pkt);
            }
        }

        bool read_synapseRouteIn = false;
        read_synapseRouteIn = SynapseRouteIn.read_nb(pkt);
        if(read_synapseRouteIn) {
            if(pkt.dest == id) {
                last_packet[pkt.id] = pkt.last;
                SynapseStreamOut.write(pkt);
            }
            else {
                SynapseRouteOut.write(pkt);
            }
        }
        if(last_packet[0] && last_packet[1] && last_packet[2] && last_packet[3]) {
            stream512u_t pkt;
            pkt.data = 0;
            pkt.last = 1;
            pkt.id = id;
            pkt.dest = id;
            SynapseStreamOut.write(pkt);
            while_loop = false;
        }
    }
}
