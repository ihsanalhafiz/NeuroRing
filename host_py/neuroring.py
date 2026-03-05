#!/usr/bin/env python3
import os
import sys
import uuid
import re
import numpy as np
import nest
import network
from network_params import default_net_dict as net_dict
from sim_params import default_sim_dict as sim_dict
from stimulus_params import default_stim_dict as stim_dict
import matplotlib.pyplot as plt
import struct
from collections import defaultdict
import math
from functools import reduce
import struct

import pyxrt
from utils_binding import *   # provides .index and .bitstreamFile

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

class NeuroRingKernel:
    def __init__(self, simulation_time, amount_of_cores, neuron_start, neuron_total, core_id, neuron_per_cu, synapse_total_per_cu, record_status, param_dict):
        self.simulation_time = simulation_time
        self.amount_of_cores = int(amount_of_cores)
        self.neuron_start = int(neuron_start)
        self.neuron_total = int(neuron_total)
        self.device = None
        self.xclbin = None
        self.uuid = None
        self.kernel_name = None
        self.kernel = None
        self.neuron_per_cu = neuron_per_cu
        self.synapse_total_per_cu = synapse_total_per_cu
        self.param_dict = param_dict
        self.record_status = record_status
        
        # Persistent BO and layout info
        self.synapseListHandle = None
        self.header_words = int(100000 * (self.neuron_per_cu/32))  # recorder words (up to 120k timesteps * 128 words/tick)
        self.header_bytes = self.header_words * 4
        self.tail_words_capacity = int(self.neuron_per_cu) * int(self.synapse_total_per_cu*2)
        self.tail_bytes_capacity = self.tail_words_capacity * 4
        self.bo_size = self.header_bytes + self.tail_bytes_capacity
        self.core_id = core_id
        
        # print all the attributes
        print(self.__dict__)

    def initialize_kernel(self, device, xclbin, uuid, kernel_neuroring, kernel_synapserouter):
        self.device = device
        self.xclbin = xclbin
        self.uuid = uuid
        # Initialize the kernel object from pyxrt
        self.kernel_neuroring = pyxrt.kernel(device, uuid, kernel_neuroring, pyxrt.kernel.shared)
        self.kernel_synapserouter = pyxrt.kernel(device, uuid, kernel_synapserouter, pyxrt.kernel.shared)
        print(f"Initialized kernel {kernel_neuroring} and {kernel_synapserouter} on device {device}")
        
        # Allocate persistent BO once per kernel and keep it for reuse across runs
        if self.tail_bytes_capacity < 0:
            self.tail_bytes_capacity = 0
        self.total_bo_size = self.header_bytes + self.tail_bytes_capacity
        self.synapseListHandle = pyxrt.bo(self.device, self.total_bo_size, pyxrt.bo.normal, self.kernel_neuroring.group_id(0))
        self.recorderMap = self.synapseListHandle.map()
        print(f"Allocated BO of {self.total_bo_size} bytes (header {self.header_bytes}, tail {self.tail_bytes_capacity})")
        
    def upload_synapse_list(self, synapse_list_data):
        self.synapseListHandle.write(synapse_list_data, 0)
        self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, synapse_list_data.nbytes, 0)
        print(f"Uploaded synapse list to {self.kernel_neuroring}")
        
    def run_neuroring(self, simulation_time):
        V_decay = np.exp(-self.param_dict['dt']/self.param_dict['tau_m'])
        I_decay = np.exp(-self.param_dict['dt']/self.param_dict['tau_syn'])
        syn_to_vm = (1.0 / self.param_dict['C_m']) * ((I_decay - V_decay) / ((1.0 / self.param_dict['tau_m']) - (1.0 / self.param_dict['tau_syn'])))
        bias_to_vm = (self.param_dict['tau_m'] / self.param_dict['C_m']) * (1.0 - V_decay)  # mV per pA
        t_ref_steps = self.param_dict['t_ref_steps']        # round(2.0/0.1)
        V_th_rel = self.param_dict['V_th_abs'] - self.param_dict['E_L']
        V_reset_rel = self.param_dict['V_reset_abs'] - self.param_dict['E_L']
        
        V_decay_bits = struct.unpack('!I', struct.pack('!f', V_decay))[0]
        I_decay_bits = struct.unpack('!I', struct.pack('!f', I_decay))[0]
        syn_to_vm_bits = struct.unpack('!I', struct.pack('!f', syn_to_vm))[0]
        bias_to_vm_bits = struct.unpack('!I', struct.pack('!f', bias_to_vm))[0]
        V_th_rel_bits = struct.unpack('!I', struct.pack('!f', V_th_rel))[0]
        V_reset_rel_bits = struct.unpack('!I', struct.pack('!f', V_reset_rel))[0]
        # convert E_L to negative value to match the hardware design but still keep the same value
        E_L_bits = struct.unpack('!I', struct.pack('!f', -self.param_dict['E_L']))[0]

         ### run the kernel
        self.runNeuroRing = self.kernel_neuroring(self.synapseListHandle, self.neuron_start, self.neuron_total,
                                                     simulation_time, 1, self.core_id, self.amount_of_cores,
                                                     V_decay_bits, I_decay_bits, syn_to_vm_bits, bias_to_vm_bits, 
                                                     V_th_rel_bits, V_reset_rel_bits, E_L_bits,t_ref_steps)
        print(f"Running kernel {self.kernel_neuroring}")

    def run_synapserouter(self, simulation_time):
        self.runSynapseRouter = self.kernel_synapserouter(simulation_time, self.amount_of_cores, self.neuron_start, self.core_id)
        print(f"Running kernel {self.kernel_synapserouter}")

    def wait_for_kernel(self):
        self.runNeuroRing.wait()
        print(f"Kernel run complete {self.kernel_neuroring}")
        self.runSynapseRouter.wait()
        print(f"Kernel run complete {self.kernel_synapserouter}")
        # Allow run handles to be GC'd between runs
        self.runNeuroRing = None
        self.runSynapseRouter = None
    
    def wait_for_synapserouter(self):
        self.runSynapseRouter.wait()
        print(f"Synapse router run complete {self.kernel_synapserouter}")

    def wait_for_neuroring(self):
        self.runNeuroRing.wait()
        print(f"Kernel run complete {self.kernel_neuroring}")

    def get_spike_recorder_array(self, sim_time):
        # Read back 128 words per timestep (4096 bits)
        size_bytes = sim_time * int(self.neuron_per_cu/32) * 4
        if size_bytes <= 0:
            return np.array([], dtype=np.uint32)
        self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size_bytes, 0)
        # Ensure a C-contiguous buffer to avoid BufferError from non-contiguous memoryviews
        buf_bytes = bytes(self.recorderMap[:size_bytes])
        return np.frombuffer(buf_bytes, dtype=np.uint32, count=sim_time * int(self.neuron_per_cu/32))

##############################################################
# Host class
##############################################################

class NeuroRingHost:
    def __init__(self, net, neuron_per_cu, synapse_total_per_cu, num_compute_units, num_fpgas, param_dict, record_status, bitstream_file):
        self.net = net
        self.param_dict = param_dict
        self.num_compute_units = num_compute_units
        self.num_fpgas = num_fpgas
        self.bitstream_file = bitstream_file
        self.devices = []
        self.xclbins = []
        self.uuids = []
        self.kernels = []  # List of NeuroRingKernel instances
        self.kernels_per_fpga = []  # List of lists: kernels assigned to each FPGA
        self.spikeRecorder_array = []
        self.record_status = record_status
        
        # Precompute DC amplitude per population and convert to IEEE-754 bits
        self.dc_amp = net.DC_amp 
        self.dc_amp_bits_per_pop = np.array(self.dc_amp, dtype='<f4').view('<u4').astype(np.uint32)

        # --- Extract synapse_list and packed_list ---
        print("Extracting synapse information...")
        
        last_pop = net.pops[-1]
        last_neuron = last_pop[-1].global_id
        self.neuron_per_cu = neuron_per_cu
        self.synapse_total_per_cu = synapse_total_per_cu
        spike_total = 100000 * int(neuron_per_cu/32)
        self.total_neurons = last_neuron
        
        max_synapse_per_neuron = 0
        
        self.synapse_data = np.zeros((self.total_neurons * self.synapse_total_per_cu, 3), dtype=np.float32)
        self.synapse_fpga = []
        # structure of synapse_data is [target, delay, weight] each of them is 32 bit float
        # each 5000 list is for one neuron. in the array 0, it consist of info [synapse_total, dc_amp, v_m]
        
        # Try to load synapse data from file first
        synapse_file = f"syndata_total{self.total_neurons}_NperCU{neuron_per_cu}_SperCU{synapse_total_per_cu}.npy"
        if os.path.exists(synapse_file):
            print(f"Loading synapse data from {synapse_file}")
            self.synapse_data = np.load(synapse_file)
        else:
            print("Synapse data file not found. Generating new synapse data...")
            for j, source_pop in enumerate(net.pops):
                for i, neuron in enumerate(source_pop):
                    neuron_idx = neuron.global_id
                    connections = nest.GetConnections(source=neuron)
                    if len(connections) > max_synapse_per_neuron:
                        max_synapse_per_neuron = len(connections)
                    self.synapse_data[(neuron_idx-1) * self.synapse_total_per_cu] = [len(connections), self.dc_amp[j], source_pop.get('V_m')[i]]
                    print(f"Processing neuron {neuron_idx} ({neuron_idx/self.total_neurons*100:.1f}%) total synapse: {len(connections)} max synapse per neuron: {max_synapse_per_neuron}", end='\r', flush=True)
                    if len(connections) == 0:
                        continue
                    cu_start = ((neuron_idx - 1) // neuron_per_cu) * neuron_per_cu + 1
                    conn_status = nest.GetStatus(connections, ['target', 'weight', 'delay'])
                    conn_info = np.array(sorted(conn_status, key=lambda x: ((int(x[0]) - cu_start) % self.total_neurons)))
                    targets = conn_info[:,0].astype(np.int32)
                    weights = conn_info[:,1].astype(np.float32) 
                    delays = (conn_info[:,2] * 10).astype(np.int32)
                    
                    idx = (neuron_idx-1) * self.synapse_total_per_cu + np.arange(1, len(conn_info) + 1)
                    self.synapse_data[idx] = np.column_stack((targets, delays, weights))
            
            # save synapse_data to numpy file with name include total_neurons
            print(f"\nSaving synapse data to {synapse_file}")
            np.save(synapse_file, self.synapse_data)
        
        synapse_fpga_file = f"synfpga_total{self.total_neurons}_NperCU{neuron_per_cu}_SperCU{synapse_total_per_cu}.npy"
        if os.path.exists(synapse_fpga_file):
            print(f"Loading synapse FPGA data from {synapse_fpga_file}")
            # Allow object arrays (list of arrays) saved via numpy
            self.synapse_fpga = np.load(synapse_fpga_file, allow_pickle=True)
            # Normalize to a list of ndarrays for downstream write calls
            if isinstance(self.synapse_fpga, np.ndarray) and self.synapse_fpga.dtype == object:
                self.synapse_fpga = list(self.synapse_fpga)
        else:
            print("Synapse FPGA data file not found. Generating new synapse FPGA data...")
            synapse_fpga_buf = np.zeros(int(((self.synapse_total_per_cu*2)*neuron_per_cu) + spike_total), dtype=np.uint32)
            for i in range(self.total_neurons):
                idx = (i*self.synapse_total_per_cu)
                idx_fpga = int(spike_total + ((i%neuron_per_cu)*(self.synapse_total_per_cu*2)))
                synapse_fpga_buf[idx_fpga] = self.synapse_data[idx][0]
                synapse_fpga_buf[idx_fpga + 1] = struct.unpack('<I', struct.pack('<f', float(self.synapse_data[idx][1])))[0]
                synapse_fpga_buf[idx_fpga + 2] = struct.unpack('<I', struct.pack('<f', float(self.synapse_data[idx][2])))[0]   
                # First pass: populate connection entries and count per-delay
                for j in range(int(np.minimum(self.synapse_data[idx][0], (self.synapse_total_per_cu-16)))):
                    target = (np.uint32(self.synapse_data[idx+j+1][0]) << 8) & 0xFFFFFF00
                    delay = np.uint32(self.synapse_data[idx+j+1][1]) & 0xFF
                    weight = struct.unpack('<I', struct.pack('<f', float(self.synapse_data[idx+j+1][2])))[0]
                    synapse_fpga_buf[idx_fpga +(j*2) + 8] = target | delay
                    synapse_fpga_buf[idx_fpga +(j*2) + 9] = weight
                
                #print progress
                print(f"Processing FPGA buffer {i} ({i/self.total_neurons*100:.1f}%)", end='\r', flush=True)
                if ((i+1)%neuron_per_cu == 0) or (i == self.total_neurons-1):
                    self.synapse_fpga.append(synapse_fpga_buf)
                    synapse_fpga_buf = np.zeros(int(((self.synapse_total_per_cu*2)*neuron_per_cu) + spike_total), dtype=np.uint32)
            print(f"\nSaving synapse FPGA data to {synapse_fpga_file}")
            np.save(synapse_fpga_file, self.synapse_fpga)
            
        self.calculate_kernel_neuron_ranges_per_fpga()
            
    def calculate_kernel_neuron_ranges_per_fpga(self):
        """
        Calculate neuron ranges for each kernel on each FPGA.
        Distributes compute units evenly across FPGAs, then assigns neurons to each kernel.
        """
        # Calculate how many kernels per FPGA (distribute as evenly as possible)
        base = self.num_compute_units // self.num_fpgas
        extra = self.num_compute_units % self.num_fpgas
        kernel_counts = [base + (1 if i < extra else 0) for i in range(self.num_fpgas)]
        
        self.kernel_neuron_ranges_per_fpga = []
        neurons_per_kernel = self.neuron_per_cu  # Maximum neurons per compute unit
        global_kernel_id = 0
        
        print(f"\nDistributing {self.total_neurons} neurons across {self.num_compute_units} compute units on {self.num_fpgas} FPGAs:")
        print(f"Kernels per FPGA: {kernel_counts}")
        
        for fpga_idx in range(self.num_fpgas):
            fpga_kernel_ranges = []
            remaining_neurons = self.total_neurons - (global_kernel_id * neurons_per_kernel)
            
            print(f"\nFPGA {fpga_idx} (Kernels: {kernel_counts[fpga_idx]}):")
            
            for local_kernel_id in range(kernel_counts[fpga_idx]):
                if remaining_neurons > 0:
                    neuron_start = global_kernel_id * neurons_per_kernel + 1
                    neuron_total = min(neurons_per_kernel, remaining_neurons)
                    remaining_neurons -= neuron_total
                else:
                    neuron_start = 0
                    neuron_total = 0
                
                fpga_kernel_ranges.append((neuron_start, neuron_total))
                print(f"  Kernel {local_kernel_id} (Global ID: {global_kernel_id}): neurons {neuron_start} to {neuron_start + neuron_total - 1 if neuron_total > 0 else 0} (total: {neuron_total})")
                global_kernel_id += 1
            
            self.kernel_neuron_ranges_per_fpga.append(fpga_kernel_ranges)
        
        print(f"\nTotal neurons assigned: {sum(sum(ranges[1] for ranges in fpga_ranges) for fpga_ranges in self.kernel_neuron_ranges_per_fpga)}")

    def initialize_devices(self):
        """
        Initialize FPGA devices, load the xclbin to each device, and initialize kernels on each device.
        """
        # Calculate how many kernels per FPGA (distribute as evenly as possible)
        base = self.num_compute_units // self.num_fpgas
        extra = self.num_compute_units % self.num_fpgas
        kernel_counts = [base + (1 if i < extra else 0) for i in range(self.num_fpgas)]
        total_cores = sum(kernel_counts)
        kernel_id = 0
        self.kernels_per_fpga = []
        for fpga_idx in range(self.num_fpgas):
            kernel_id = 0
            core_offset = sum(kernel_counts[:fpga_idx])
            device = pyxrt.device(fpga_idx)
            xclbin = pyxrt.xclbin(self.bitstream_file)
            uuid = device.load_xclbin(xclbin)
            self.devices.append(device)
            self.xclbins.append(xclbin)
            self.uuids.append(uuid)
            print(f"Initialized device {fpga_idx} with XCLBIN UUID: {uuid.to_string()}")
            fpga_kernels = []
            for k in range(kernel_counts[fpga_idx]):
                kernel_name = f"NeuroRing:{{NeuroRing_{kernel_id}}}"  # Correct format
                kernel_synapserouter = f"SynapseRouter:{{SynapseRouter_{kernel_id}}}"
                # Use neuron_start and neuron_total from self.kernel_neuron_ranges
                neuron_start, neuron_total = self.kernel_neuron_ranges_per_fpga[fpga_idx][kernel_id]
                kernel = NeuroRingKernel(
                    simulation_time=1,
                    amount_of_cores = total_cores,
                    neuron_start=neuron_start,
                    neuron_total=neuron_total,
                    core_id=(kernel_id + core_offset),
                    neuron_per_cu=self.neuron_per_cu,
                    synapse_total_per_cu=self.synapse_total_per_cu,
                    record_status=self.record_status,
                    param_dict=self.param_dict,
                )
                kernel.initialize_kernel(device, xclbin, uuid, kernel_name, kernel_synapserouter)
                self.kernels.append(kernel)
                fpga_kernels.append(kernel)
                kernel_id += 1
            self.kernels_per_fpga.append(fpga_kernels)

    def run_kernels(self, sim_time):
        ##for fpga_idx in range(self.num_fpgas):
        ##    for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
        ##        
        for fpga_idx in range(self.num_fpgas):
            for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
                print(f"running FPGA {fpga_idx} Kernel {i}")
                # Ensure we pass a contiguous uint32 buffer
                buf = np.asarray(self.synapse_fpga[i], dtype=np.uint32, order='C')
                kernel.run_kernel(buf, sim_time)
    
    def wait_for_kernels(self, sim_time):
        for fpga_idx in range(self.num_fpgas):
            for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
                print(f"waiting for FPGA {fpga_idx} Kernel {i}")
                kernel.wait_for_kernel(sim_time)
                
    def get_spike_recorder_array(self, sim_time):
        # Vectorized decoding: iterate bits (0..31) only; avoid triple nested loops
        self.neuronidx = []
        self.spikeidx = []

        spike_times_chunks = []
        neuron_ids_chunks = []

        for fpga_idx in range(self.num_fpgas):
            for _, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
                array_spike = kernel.get_spike_recorder_array(sim_time)
                if len(array_spike) == 0:
                    continue

                # Shape to [time, 128 words] where each word encodes 32 neurons
                # Be robust to buffers that contain more or fewer ticks than requested
                words = np.asarray(array_spike, dtype=np.uint32)
                total_words = int(len(words))
                if total_words == 0:
                    continue
                available_ticks = total_words // int(self.neuron_per_cu/32)
                if available_ticks == 0:
                    continue
                n_ticks = min(sim_time, available_ticks)
                arr2d = words[: n_ticks * int(self.neuron_per_cu/32)].reshape(n_ticks, int(self.neuron_per_cu/32))

                # Map local neuron indices (0..2047) to global IDs for this kernel
                neuron_start = getattr(kernel, 'neuron_start', 0)

                # Process 32 bits with vectorized bit masks
                for bit in range(32):
                    mask_hits = ((arr2d >> bit) & 1).astype(bool)
                    if not mask_hits.any():
                        continue
                    t_idx, j_idx = np.nonzero(mask_hits)
                    local_ids = j_idx * 32 + bit
                    global_ids = neuron_start + local_ids
                    spike_times_chunks.append(t_idx.astype(np.int64, copy=False))
                    neuron_ids_chunks.append(global_ids.astype(np.int64, copy=False))

        if spike_times_chunks:
            self.spikeidx = np.concatenate(spike_times_chunks).tolist()
            self.neuronidx = np.concatenate(neuron_ids_chunks).tolist()
        else:
            self.spikeidx = []
            self.neuronidx = []

        return self.spikeidx, self.neuronidx
    
    def plot_spike_recorder_array(self, filename="save.png", start_tick=None, end_tick=None, show=False, figsize=(8, 4), dpi=300, s=1):
        return self._plot_spike_recorder_array_impl(filename=filename, start_tick=start_tick, end_tick=end_tick, show=show, figsize=figsize, dpi=dpi, s=s)

    def _plot_spike_recorder_array_impl(self, filename="save.png", start_tick=None, end_tick=None, show=False, figsize=(8, 4), dpi=300, s=1):
        # Prepare data and optionally filter by time range
        spikes = np.asarray(self.spikeidx, dtype=np.int64) if hasattr(self, 'spikeidx') else np.array([], dtype=np.int64)
        neurons = np.asarray(self.neuronidx, dtype=np.int64) if hasattr(self, 'neuronidx') else np.array([], dtype=np.int64)

        if spikes.size and (start_tick is not None or end_tick is not None):
            st = start_tick if start_tick is not None else int(spikes.min())
            en = end_tick if end_tick is not None else int(spikes.max())
            if en < st:
                st, en = en, st
            mask = (spikes >= st) & (spikes <= en)
            spikes = spikes[mask]
            neurons = neurons[mask]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(spikes, neurons, s=s, rasterized=True)
        ax.set_xlabel("time (ticks)")
        ax.set_ylabel("neuron id")
        fig.tight_layout()

        if filename is not None:
            dirpath = os.path.dirname(filename)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            if not show:
                plt.close(fig)
            return filename

        if show:
            plt.show()
        else:
            plt.close(fig)
        return None

