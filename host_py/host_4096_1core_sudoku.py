import logging
import pickle
import time

import matplotlib.pyplot as plt
import nest
import numpy as np
import sudoku_net
from helpers_sudoku import get_puzzle, plot_field, validate_solution

import neuroring_sudoku
from utils_binding import *   # provides .index and .bitstreamFile
import pyxrt
import re


nest.SetKernelStatus({"local_num_threads": 8})
nest.set_verbosity("M_WARNING")
logging.basicConfig(level=logging.INFO)

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


def reset_spike():
    host.kernels_per_fpga[0][0].upload_synapse_list(host.synapse_fpga[0])
    host.kernels_per_fpga[0][1].upload_synapse_list(host.noise_stim_fpga)

def get_power(sim_time_start, sim_time_end):
    start_time = time.time()
    sim_time = sim_time_end - sim_time_start
    host.kernels_per_fpga[0][0].run_neuroring(sim_time_start, sim_time_end)
    host.kernels_per_fpga[0][1].run_poisson(sim_time)

    host.kernels_per_fpga[0][0].run_synapserouter(sim_time)
    host.kernels_per_fpga[0][1].run_synapserouter(sim_time)
    
    time.sleep(0.01)
    power = measure_total_power()

    host.kernels_per_fpga[0][0].wait_for_kernel()
    host.kernels_per_fpga[0][1].wait_for_kernel()
    end_time = time.time()
    return power

def simulate(sim_time_start, sim_time_end):
    start_time = time.time()
    sim_time = sim_time_end - sim_time_start
    host.kernels_per_fpga[0][0].run_neuroring(sim_time_start, sim_time_end)
    host.kernels_per_fpga[0][1].run_poisson(sim_time)

    host.kernels_per_fpga[0][0].run_synapserouter(sim_time)
    host.kernels_per_fpga[0][1].run_synapserouter(sim_time)
    
    host.kernels_per_fpga[0][0].wait_for_kernel()
    host.kernels_per_fpga[0][1].wait_for_kernel()
    end_time = time.time()
    return (end_time - start_time)

def get_spike_poisson(sim_time):
    spikeidx, neuronidx = host.get_spike_recorder_array(sim_time)
    # Pair and sort based on neuronidx (low to high)
    paired = sorted(zip(spikeidx, neuronidx), key=lambda x: x[1])
    # Filter only pairs where neuronidx >= 3646
    paired = [p for p in paired if p[1] >= 3646]
    return paired

def get_spike(sim_time, n_neurons=3645, group_size=5):
    spikeidx, neuronidx = host.get_spike_recorder_array(sim_time)

    spikeidx = np.asarray(spikeidx)
    neuronidx = np.asarray(neuronidx)

    # Keep only the Sudoku population (neurons 1..n_neurons)
    mask = (neuronidx >= 1) & (neuronidx <= n_neurons)
    spikeidx = spikeidx[mask]
    neuronidx = neuronidx[mask]

    n_groups = (n_neurons + group_size - 1) // group_size
    senders = [[] for _ in range(n_groups)]
    times = [[] for _ in range(n_groups)]

    # Group by blocks of `group_size` neurons: (1..5), (6..10), ...
    for t, n in zip(spikeidx, neuronidx):
        gi = (int(n) - 1) // group_size
        if 0 <= gi < n_groups:
            senders[gi].append(int(n))
            times[gi].append(float(t))

    # Match the common `spiketrains` shape: list (len=729) of [ {'senders','times'} ]
    spiketrains = []
    for i in range(n_groups):
        if times[i]:
            order = np.argsort(times[i])
            s = np.asarray([senders[i][j] for j in order], dtype=np.int32)
            tt = np.asarray([times[i][j] for j in order], dtype=np.float32)
        else:
            s = np.asarray([], dtype=np.int32)
            tt = np.asarray([], dtype=np.float32)
        spiketrains.append([{'senders': s, 'times': tt}])

    return spiketrains


def plot(filename="spike_recorder_sudoku.png", start_tick=0, end_tick=100):
    host.plot_spike_recorder_array(filename, start_tick, end_tick)
    np.savetxt("spikeidx_sudoku.csv", host.spikeidx, delimiter=",")
    np.savetxt("neuronidx_sudoku.csv", host.neuronidx, delimiter=",")


print("================================================")
print("Starting Sudoku solver Puzzle")
print("================================================")

puzzle_index = 15
noise_rate = 200
sim_time = 100
stim_rate = 200
max_sim_time = 10000
max_iterations = max_sim_time // sim_time

puzzle = get_puzzle(puzzle_index)
network = sudoku_net.SudokuNet(pop_size=5, input=puzzle, noise_rate=noise_rate, stim_rate=stim_rate)

solution_states = np.zeros((max_iterations, 9, 9), dtype=np.int_)

param_dict = {
    'dt': 0.1,
    'tau_m': 20.0,
    'tau_syn': 5.0,
    'C_m': 250.0,
    'E_L': -65.0,
    't_ref_steps': 20,
    'V_th_abs': -50.0,
    'V_reset_abs': -70.0,
}
record_status = 1
host = neuroring_sudoku.NeuroRingHost(network, 4096, 7000, 1, 1, param_dict, record_status, "/home/miahafiz/NeuroRing/_build_dir.hw.NUM_4096.CORE_10.FREQ_300/krnl_neuroring_hw.xclbin")
host.initialize_devices()
print("Initialized devices")

num_runs = 20
times_execution = []
times_all = []
power_readings = []
sim_time_start = 0
sim_time_end = 5000

for run in range(num_runs):
    print(f"Running run {run + 1} of {num_runs}")
    print("--------------------------------")
    start_time = time.time()
    reset_spike()
    exec_time = simulate(sim_time_start, sim_time_end)
    times_execution.append(exec_time)
    print(f"simulation execution: {exec_time}")
    paired_spike = get_spike(sim_time_end)

    # Initialize solution as a copy of the puzzle to keep track of placed digits
    solution = puzzle.copy()
    solution_states = {} # Assuming this is a dict or array initialized earlier
    run = 0

    for row in range(9):
        for col in range(9):
            # obtain indices of the spike recorders coding for digits in
            # the current cell
            spike_recorders = network.io_indices[row, col]
            # spiketrains for all digits in the current cells
            idx = np.asarray(spike_recorders, dtype=np.int64).ravel()
            cell_spikes = [paired_spike[int(i)] for i in idx]
            spike_counts = np.array([len(s[0]["times"]) for s in cell_spikes])
            #print(spike_counts)
            # if two digits have the same activation, pick one at random
            winning_digit = int(np.random.choice(np.flatnonzero(spike_counts == spike_counts.max()))) + 1
            #print(winning_digit)
            #print("-------------------------------------------------------------------------------")
            solution[row, col] = winning_digit

    # Save and validate
    solution_states[run] = solution
    valid, cells, rows, cols = validate_solution(puzzle, solution)
    end_time = time.time()
    ratio_correct = (np.sum(cells) + np.sum(rows) + np.sum(cols)) / 27
    times_all.append(end_time - start_time)
    print(f"time execution: {end_time - start_time}")
    print(f"performance: {np.round(ratio_correct, 3)} cell:{np.sum(cells)} rows:{np.sum(rows)} cols:{np.sum(cols)}")
    print(puzzle)
    print("--------------------------------")
    print(solution)
    print("--------------------------------")
    print(f"valid: {valid}")
    power = get_power(sim_time_start, sim_time_end)
    power_readings.append(power)
    print(f"power: {power}")
    print("--------------------------------")
    print("")
    
# Average the times and power readings
average_times_execution = np.mean(times_execution)
average_times_all = np.mean(times_all)
average_power = np.mean(power_readings)
print(f"Average time execution: {average_times_execution}")
print(f"Average time all: {average_times_all}")
print(f"Average power: {average_power}")

print("================================================")
print("Finished Sudoku solver Puzzle")
print("================================================")
print("")
print("")
print("")