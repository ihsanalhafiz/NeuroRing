import time
import nest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neo
import quantities as pq
import re
from elephant.statistics import isi, cv, mean_firing_rate
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
from pathlib import Path

import subprocess

POWER_DEVICES = ["0000:2a:00.1"]  # only one U55C device

def measure_board_power(device):
    """Return power (W) for a single board using xrt-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            ["xrt-smi", "examine", "-d", device, "-r", "electrical"],
            text=True
        )
    except Exception as e:
        print(f"Failed to read power for {device}: {e}")
        return None

    m = re.search(r"^\s*Power\s+:\s*([\d.]+)\s*Watts", out, re.MULTILINE)
    if not m:
        print(f"Could not parse power for {device}")
        return None
    return float(m.group(1))

def measure_total_power():
    """Sum power over all devices in POWER_DEVICES."""
    readings = [measure_board_power(dev) for dev in POWER_DEVICES]
    readings = [p for p in readings if p is not None]
    return sum(readings) if readings else None

## import model implementation
import network
## import (default) parameters (network, simulation, stimulus)
from network_params import default_net_dict as net_dict
from sim_params import default_sim_dict as sim_dict
from stimulus_params import default_stim_dict as stim_dict

# Import library NeuroRing for FPGA and pyxrt
import neuroring
import pyxrt
from utils_binding import *   # provides .index and .bitstreamFile

# Create network and connect neurons
net = network.Network(sim_dict, net_dict, stim_dict)
net.create()
net.connect()

print(net.pops)

param_dict = {
    'dt': 0.1,
    'tau_m': 10.0,
    'tau_syn': 0.5,
    'C_m': 250.0,
    'E_L': -65.0,
    't_ref_steps': 20,
    'V_th_abs': -50.0,
    'V_reset_abs': -65.0,
}
# 1 for recording, 0 for not recording spike
record_status = 0

host = neuroring.NeuroRingHost(net, 8192, 7000, 5, 1, param_dict, record_status, "/home/miahafiz/NeuroRing/_build_dir.hw.NUM_8192.CORE_5.FREQ_300/krnl_neuroring_hw.xclbin")

host.initialize_devices()
print("Initialized devices")

host.kernels_per_fpga[0][0].upload_synapse_list(host.synapse_fpga[0])
host.kernels_per_fpga[0][1].upload_synapse_list(host.synapse_fpga[1])
host.kernels_per_fpga[0][2].upload_synapse_list(host.synapse_fpga[2])
host.kernels_per_fpga[0][3].upload_synapse_list(host.synapse_fpga[3])
host.kernels_per_fpga[0][4].upload_synapse_list(host.synapse_fpga[4])

print("Uploaded synapse list to FPGA")

# Run simulation multiple times and measure time
num_runs = 20
times = []
power_readings = []
# timestep is the number of steps in the simulation
timestep = 100000

for run in range(num_runs):
    print(f"Running simulation {run + 1}/{num_runs}")
    start_time = time.time()
    host.kernels_per_fpga[0][0].run_neuroring(timestep)
    host.kernels_per_fpga[0][0].run_synapserouter(timestep)
    host.kernels_per_fpga[0][1].run_neuroring(timestep)
    host.kernels_per_fpga[0][1].run_synapserouter(timestep)
    host.kernels_per_fpga[0][2].run_neuroring(timestep)
    host.kernels_per_fpga[0][2].run_synapserouter(timestep)
    host.kernels_per_fpga[0][3].run_neuroring(timestep)
    host.kernels_per_fpga[0][3].run_synapserouter(timestep)
    host.kernels_per_fpga[0][4].run_neuroring(timestep)
    host.kernels_per_fpga[0][4].run_synapserouter(timestep)
    
    # wait 1 second and measure power while kernels are running
    # because of waiting for kernel, sleep will not disturb the simulation time
    time.sleep(1.0)
    power = measure_total_power()
    if power is not None:
        print(f"Measured total power: {power:.3f} W")
        power_readings.append(power)

    host.kernels_per_fpga[0][0].wait_for_kernel()
    host.kernels_per_fpga[0][1].wait_for_kernel()
    host.kernels_per_fpga[0][2].wait_for_kernel()
    host.kernels_per_fpga[0][3].wait_for_kernel()
    host.kernels_per_fpga[0][4].wait_for_kernel()

    end_time = time.time()
    duration = end_time - start_time
    times.append(duration)
    print(f"Time taken for run {run + 1}: {duration} seconds")

avg_time = sum(times) / len(times)
print(f"Average time over {num_runs} runs: {avg_time} seconds")

if power_readings:
    avg_power = sum(power_readings) / len(power_readings)
    print(f"Average total power over {num_runs} runs: {avg_power:.3f} W")