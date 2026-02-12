import time

import nest
import network
import numpy as np
import neuroring
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict

from utils_binding import *   # provides .index and .bitstreamFile
import pyxrt

net = network.Network(sim_dict, net_dict, stim_dict)
net.create()
net.connect()

host = neuroring.NeuroRingHost(net, 10, 1, "/home/miahafiz/NeuroRing_v3/build_dir.hw.xilinx_u55c_gen3x16_xdma_3_202210_1/neuroringcore.xclbin")

host.initialize_devices()
print("Initialized devices")

# ---------------------
# Pre-run verification
# ---------------------
# Write synapse buffers into each kernel's BO, then verify headers and delay buckets
#for i, kernel in enumerate(host.kernels_per_fpga[0]):
#    print(f"Priming BO for kernel {i} with synapse buffer")
#    buf = np.asarray(host.synapse_fpga[i], dtype=np.uint32, order='C')
#    kernel.synapseListHandle.write(buf, 0)
#    kernel.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, buf.nbytes, 0)

# Verify a few neurons on kernel 0
#print("\nVerification: kernel 0, local neuron 0 header preview")
#info0 = host.verify_kernel_neuron(fpga_idx=0, kernel_idx=0, local_neuron_index=0, pairs_preview=8)
#print("I_bias_bits, V_m_bits:", info0.get("I_bias_bits"), info0.get("V_m_bits"))
#print("delay_meta[0..7]:", info0.get("delay_meta_u32")[:8])
#print("first 8 words after 72:", info0.get("pairs_u32"))
#
#print("\nVerification: kernel 0, local neuron 0 delay bucket 1")
#pairs_d1 = host.verify_kernel_delay(fpga_idx=0, kernel_idx=0, local_neuron_index=0, delay_value=1)
#print("delay=1 pairs shape:", pairs_d1.shape)
#print("first 4 pairs (target|delay, weight):", pairs_d1[:4])

# ---------------------
# Run simulation
# ---------------------
timestep = 100000

start_time = time.time()
host.kernels_per_fpga[0][0].upload_synapse_list(host.synapse_fpga[0])
host.kernels_per_fpga[0][1].upload_synapse_list(host.synapse_fpga[1])
host.kernels_per_fpga[0][2].upload_synapse_list(host.synapse_fpga[2])
host.kernels_per_fpga[0][3].upload_synapse_list(host.synapse_fpga[3])
host.kernels_per_fpga[0][4].upload_synapse_list(host.synapse_fpga[4])
host.kernels_per_fpga[0][5].upload_synapse_list(host.synapse_fpga[5])
host.kernels_per_fpga[0][6].upload_synapse_list(host.synapse_fpga[6])
host.kernels_per_fpga[0][7].upload_synapse_list(host.synapse_fpga[7])
host.kernels_per_fpga[0][8].upload_synapse_list(host.synapse_fpga[8])
host.kernels_per_fpga[0][9].upload_synapse_list(host.synapse_fpga[9])

host.kernels_per_fpga[0][0].run_axon_loader(timestep)
host.kernels_per_fpga[0][1].run_axon_loader(timestep)
host.kernels_per_fpga[0][2].run_axon_loader(timestep)
host.kernels_per_fpga[0][3].run_axon_loader(timestep)
host.kernels_per_fpga[0][4].run_axon_loader(timestep)
host.kernels_per_fpga[0][5].run_axon_loader(timestep)
host.kernels_per_fpga[0][6].run_axon_loader(timestep)
host.kernels_per_fpga[0][7].run_axon_loader(timestep)
host.kernels_per_fpga[0][8].run_axon_loader(timestep)
host.kernels_per_fpga[0][9].run_axon_loader(timestep)

host.kernels_per_fpga[0][0].run_neuroring(timestep)
host.kernels_per_fpga[0][1].run_neuroring(timestep)
host.kernels_per_fpga[0][2].run_neuroring(timestep)
host.kernels_per_fpga[0][3].run_neuroring(timestep)
host.kernels_per_fpga[0][4].run_neuroring(timestep)
host.kernels_per_fpga[0][5].run_neuroring(timestep)
host.kernels_per_fpga[0][6].run_neuroring(timestep)
host.kernels_per_fpga[0][7].run_neuroring(timestep)
host.kernels_per_fpga[0][8].run_neuroring(timestep)
host.kernels_per_fpga[0][9].run_neuroring(timestep)

print("Running kernels")
host.kernels_per_fpga[0][0].wait_for_axon()
host.kernels_per_fpga[0][1].wait_for_axon()
host.kernels_per_fpga[0][2].wait_for_axon()
host.kernels_per_fpga[0][3].wait_for_axon()
host.kernels_per_fpga[0][4].wait_for_axon()
host.kernels_per_fpga[0][5].wait_for_axon()
host.kernels_per_fpga[0][6].wait_for_axon()
host.kernels_per_fpga[0][7].wait_for_axon()
host.kernels_per_fpga[0][8].wait_for_axon()
host.kernels_per_fpga[0][9].wait_for_axon()

print("Waiting for axon")
host.kernels_per_fpga[0][0].wait_for_neuroring()
host.kernels_per_fpga[0][1].wait_for_neuroring()
host.kernels_per_fpga[0][2].wait_for_neuroring()
host.kernels_per_fpga[0][3].wait_for_neuroring()
host.kernels_per_fpga[0][4].wait_for_neuroring()
host.kernels_per_fpga[0][5].wait_for_neuroring()
host.kernels_per_fpga[0][6].wait_for_neuroring()
host.kernels_per_fpga[0][7].wait_for_neuroring()
host.kernels_per_fpga[0][8].wait_for_neuroring()
host.kernels_per_fpga[0][9].wait_for_neuroring()

print("Waiting for neuroring")

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

host.get_spike_recorder_array(timestep)
print("Getting spike recorder array")
host.plot_spike_recorder_array(filename="spike_recorder_281125.png", start_tick=6000, end_tick=8000)
print("Plotted spike recorder array")

# save host.spikeidx and host.neuronidx to csv
np.savetxt("spikeidx.csv", host.spikeidx, delimiter=",")
np.savetxt("neuronidx.csv", host.neuronidx, delimiter=",")
print("Saved spikeidx and neuronidx to csv")