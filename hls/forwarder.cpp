#include "hls_stream.h"
#include "ap_axi_sdata.h"

using axis_t = ap_axiu<128, /*U=*/0, /*ID=*/0, /*DEST=*/8>; // 8-bit TDEST

void forwarder(hls::stream<axis_t>& in,
            hls::stream<axis_t> out[8]) {
  #pragma HLS interface axis port=in
  #pragma HLS interface axis port=out
  #pragma HLS array_partition variable=out complete
  #pragma HLS dataflow
  #pragma HLS interface ap_ctrl_none port=return

  while (true) {
    axis_t pkt = in.read();          // blocks until data
    ap_uint<8> d = pkt.dest;         // use TDEST as destination selector [0..7]
    out[d].write(pkt);               // forward; TLAST/KEEP/etc. ride along
  }
}
