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
extern "C" void SerialOut(
    hls::stream<stream256u_t>    &SynapseIn,
    hls::stream<ap_axiu<256, 0, 0, 0>>& data_output)
{
    #pragma HLS INTERFACE axis port=SynapseIn bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=data_output bundle=AXIS_OUT
    #pragma HLS INTERFACE ap_ctrl_none port=return

    while(true) {
        #pragma HLS PIPELINE II=1
        stream256u_t pkt;
        SynapseIn.read(pkt);
        data_output.write(pkt);
    }
}