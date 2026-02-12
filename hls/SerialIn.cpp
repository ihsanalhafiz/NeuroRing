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
extern "C" void SerialIn(
    hls::stream<ap_axiu<256, 0, 0, 0>>& data_input,
    hls::stream<stream256u_t>    &SynapseOut)
{
    #pragma HLS INTERFACE axis port=data_input bundle=AXIS_IN
    #pragma HLS INTERFACE axis port=SynapseOut bundle=AXIS_OUT
    #pragma HLS INTERFACE ap_ctrl_none port=return

    while(true) {
        #pragma HLS PIPELINE II=1
        ap_axiu<256, 0, 0, 0> data_out;
        data_input.read(data_out);
        SynapseOut.write(data_out);
    }
}