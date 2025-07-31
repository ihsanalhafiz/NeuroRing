import os
import sys
import pyxrt
import contextlib
from neuroring import NeuroRingHost, NeuroRingKernel
import nest
import network
import numpy as np
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
from utils_binding import Options  # Ensure this module provides .index and .bitstreamFile

net = network.Network(sim_dict, net_dict, stim_dict)
net.create()
net.connect()

print(net.pops)

host = NeuroRingHost(net, 1, 1, "/home/miahafiz/NeuroRing/krnl_neuroring_hw.xclbin")
host.initialize_devices()

host.kernels_per_fpga[0][0].run_kernel(host.packed_list_per_fpga[0][0], 1000)

host.kernels_per_fpga[0][0].wait_for_kernel()
print(host.kernels[0].spikeRecorder_array)
