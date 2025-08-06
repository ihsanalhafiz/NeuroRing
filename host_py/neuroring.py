#!/usr/bin/env python3
import os
import sys
import uuid
import re
import numpy as np
import nest
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict

import struct
from collections import defaultdict
import math

from utils_binding import *   # provides .index and .bitstreamFile
import pyxrt

class NeuroRingKernel:
    def __init__(self, simulation_time, threshold, membrane_potential, amount_of_cores, neuron_start, neuron_total, dcstim_start, dcstim_total, dcstim_amp):
        self.simulation_time = simulation_time
        threshold_float = struct.unpack('<I', struct.pack('<f', threshold))[0]  # IEEE-754 bits
        self.threshold = int(threshold_float)
        membrane_potential_float = struct.unpack('<I', struct.pack('<f', membrane_potential))[0]  # IEEE-754 bits
        self.membrane_potential = int(membrane_potential_float)
        self.amount_of_cores = int(amount_of_cores)
        self.neuron_start = int(neuron_start)
        self.neuron_total = int(neuron_total)
        self.dcstim_start = int(dcstim_start)
        self.dcstim_total = int(dcstim_total)
        dcstim_amp_float = struct.unpack('<I', struct.pack('<f', dcstim_amp))[0]  # IEEE-754 bits
        self.dcstim_amp = int(dcstim_amp_float)
        self.device = None
        self.xclbin = None
        self.uuid = None
        self.kernel_name = None
        self.kernel = None
        
        print(f"threshold: {threshold}")
        print(f"membrane_potential: {membrane_potential}")
        print(f"dcstim_amp: {dcstim_amp}")
        
        # print all the attributes
        print(self.__dict__)

    def initialize_kernel(self, device, xclbin, uuid, kernel_name, kernel_axon_loader):
        self.device = device
        self.xclbin = xclbin
        self.uuid = uuid
        self.kernel_name = kernel_name
        # Initialize the kernel object from pyxrt
        self.kernel_neuroring = pyxrt.kernel(device, uuid, kernel_name, pyxrt.kernel.shared)
        self.kernel_axon_loader = pyxrt.kernel(device, uuid, kernel_axon_loader, pyxrt.kernel.shared)
        print(f"Initialized kernel {kernel_name} and {kernel_axon_loader} on device {device}")
        
    def run_kernel(self, synapse_list_data, simulation_time):
        # copy the synapse_list_data to the synapselist_array
        self.synapseListHandle = pyxrt.bo(self.device, self.neuron_total*10000*4, pyxrt.bo.normal, self.kernel_axon_loader.group_id(0))
        self.spikeRecorderHandle = pyxrt.bo(self.device, 64*4*simulation_time, pyxrt.bo.normal, self.kernel_axon_loader.group_id(1))
        
        self.synapseListHandle.write(synapse_list_data, 0)
        self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self.neuron_total*10000*4, 0)
        spikeinput = np.zeros(64*simulation_time, dtype=np.uint32)
        #spikeinput[0] = 65535   
        self.spikeRecorderHandle.write(spikeinput, 0)
        self.spikeRecorderHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 64*4*simulation_time, 0)
        #self.spikeRecorderHandle.write(self.zeros_group1, 0)
        #self.spikeRecorderHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, int(math.ceil(self.neuron_total/32))*4, 0)
        #thr = -50.0
        #thr_bits = struct.unpack('<I', struct.pack('<f', thr))[0]  # IEEE-754 bits
        #thr_bits = np.uint32(thr_bits)                             # keep it 32-bit

        print(f"write done kernel {self.kernel_axon_loader}")
        ### run the kernel
        self.runAxonLoader = self.kernel_axon_loader(self.synapseListHandle, self.spikeRecorderHandle, self.neuron_start,
                                                     self.neuron_total, self.dcstim_start, self.dcstim_total, self.dcstim_amp,
                                                     simulation_time, 1)
        print(f"Running kernel {self.kernel_axon_loader}")
        self.runNeuroRing = self.kernel_neuroring(simulation_time, self.threshold, self.membrane_potential, self.amount_of_cores, self.neuron_start, self.neuron_total)
        print(f"Running kernel {self.kernel_neuroring}")    


    def wait_for_kernel(self, sim_time):
        self.runAxonLoader.wait()
        print(f"Axon loader run complete {self.kernel_axon_loader}")
        self.runNeuroRing.wait()
        print(f"Kernel run complete {self.kernel_neuroring}")

    
    def wait_for_axon(self):
        self.runAxonLoader.wait()
        print(f"Axon loader run complete {self.kernel_axon_loader}")

    def wait_for_neuroring(self):
        self.runNeuroRing.wait()
        print(f"Kernel run complete {self.kernel_neuroring}")
        
    def get_spike_recorder_array(self, sim_time):
        self.spikeRecorderHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 64*4*sim_time, 0)
        self.spikeRecorder_array = np.frombuffer(self.spikeRecorderHandle.read(64*4*sim_time, 0), dtype=np.uint32)
        return self.spikeRecorder_array


class NeuroRingHost:
    def __init__(self, net, num_compute_units, num_fpgas, bitstream_file):
        self.net = net
        self.num_compute_units = num_compute_units
        self.num_fpgas = num_fpgas
        self.bitstream_file = bitstream_file
        self.devices = []
        self.xclbins = []
        self.uuids = []
        self.kernels = []  # List of NeuroRingKernel instances
        self.kernels_per_fpga = []  # List of lists: kernels assigned to each FPGA
        self.spikeRecorder_array = []
        # --- Extract synapse_list and packed_list ---
        print("Extracting synapse information...")
        synapse_list = []
        for i, target_pop in enumerate(net.pops):
            for j, source_pop in enumerate(net.pops):
                if net.num_synapses[i][j] > 0:
                    connections = nest.GetConnections(source=source_pop, target=target_pop)
                    if len(connections) > 0:
                        conn_info = nest.GetStatus(connections, ['source', 'target', 'weight', 'delay'])
                        for source, target, weight, delay in conn_info:
                            synapse_list.append([source, target, int(delay*10), weight])
        synapse_list.sort(key=lambda x: x[0])
        self.synapse_list = synapse_list
        
        # 1. Group synapses by source neuron
        source_dict = defaultdict(list)
        for row in synapse_list:
            source_dict[row[0]].append(row)
        # 2. Get all unique source neurons, sorted for reproducibility
        sources = sorted(source_dict.keys())

        # 3. Calculate CUs per FPGA
        num_cus_per_fpga = [self.num_compute_units // self.num_fpgas + (1 if i < self.num_compute_units % self.num_fpgas else 0) for i in range(self.num_fpgas)]

        # 4. Split sources into groups for each CU, grouped by FPGA
        synapse_list_per_fpga = []
        cu_idx = 0
        for fpga_idx, cu_count in enumerate(num_cus_per_fpga):
            fpga_cu_synapses = []
            for cu in range(cu_count):
                # Calculate the range of sources for this CU
                start = cu_idx * len(sources) // self.num_compute_units
                end = (cu_idx + 1) * len(sources) // self.num_compute_units
                group_sources = sources[start:end]
                group_synapses = []
                for src in group_sources:
                    group_synapses.extend(source_dict[src])
                fpga_cu_synapses.append(group_synapses)
                cu_idx += 1
            synapse_list_per_fpga.append(fpga_cu_synapses)

        self.synapse_list_per_fpga = synapse_list_per_fpga
            
        # --- Create packed_list ---
        self.packed_list_per_fpga = []
        self.kernel_neuron_ranges_per_fpga = []
        block_size = 10000

        for fpga_cu_synapses in self.synapse_list_per_fpga:
            fpga_packed_lists = []
            fpga_neuron_ranges = []
            for cu_synapses in fpga_cu_synapses:
                cu_source_dict = defaultdict(list)
                for row in cu_synapses:
                    cu_source_dict[row[0]].append(row)
                if cu_source_dict:
                    min_src = min(cu_source_dict.keys())
                    max_src = max(cu_source_dict.keys())
                    neuron_start = min_src
                    neuron_total = max_src - min_src + 1
                else:
                    neuron_start = -1
                    neuron_total = 0
                fpga_neuron_ranges.append((neuron_start, neuron_total))
                cu_packed_list = [0] * (neuron_total * block_size)
                if neuron_total > 0 and neuron_start >= 0:
                    for idx, src in enumerate(range(neuron_start, neuron_start + neuron_total)):
                        syns = cu_source_dict.get(src, [])
                        block_start = idx * block_size
                        if syns:
                            cu_packed_list[block_start] = len(syns)
                            for i, (src, tgt, dly, wgt) in enumerate(syns):
                                tgt = int(tgt) & 0xFFFFFF
                                dly = int(dly) & 0xFF
                                packed_td = (tgt << 8) | dly  # target_id upper 24 bits, delay lower 8 bits
                                wgt_bits = struct.unpack('>I', struct.pack('>f', float(wgt)))[0]
                                cu_packed_list[block_start + 1 + 2*i] = packed_td
                                cu_packed_list[block_start + 2 + 2*i] = wgt_bits
                        # else: block is already zeroed
                cu_packed_list = np.array(cu_packed_list, dtype=np.uint32)
                fpga_packed_lists.append(cu_packed_list)
            self.packed_list_per_fpga.append(fpga_packed_lists)
            self.kernel_neuron_ranges_per_fpga.append(fpga_neuron_ranges)


    def initialize_devices(self):
        """
        Initialize FPGA devices, load the xclbin to each device, and initialize kernels on each device.
        """
        # Calculate how many kernels per FPGA (distribute as evenly as possible)
        base = self.num_compute_units // self.num_fpgas
        extra = self.num_compute_units % self.num_fpgas
        kernel_counts = [base + (1 if i < extra else 0) for i in range(self.num_fpgas)]
        kernel_id = 0
        self.kernels_per_fpga = []
        for fpga_idx in range(self.num_fpgas):
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
                #kernel_name = "NeuroRing"  # Correct format
                kernel_axon_loader = f"AxonLoader:{{AxonLoader_{kernel_id}}}"
                #kernel_axon_loader = "AxonLoader"
                # Use neuron_start and neuron_total from self.kernel_neuron_ranges
                neuron_start, neuron_total = self.kernel_neuron_ranges_per_fpga[fpga_idx][kernel_id]
                kernel = NeuroRingKernel(
                    simulation_time=1,
                    threshold=net_dict["neuron_params"]["V_th"],
                    membrane_potential=net_dict["neuron_params"]["V_reset"],
                    amount_of_cores = self.num_compute_units,
                    neuron_start=neuron_start,
                    neuron_total=neuron_total,
                    dcstim_start=stim_dict["dc_start"],
                    dcstim_total=stim_dict["dc_dur"],
                    dcstim_amp=np.average(self.net.DC_amp)
                )
                kernel.initialize_kernel(device, xclbin, uuid, kernel_name, kernel_axon_loader)
                self.kernels.append(kernel)
                fpga_kernels.append(kernel)
                kernel_id += 1
            self.kernels_per_fpga.append(fpga_kernels)

    def run_kernels(self, sim_time):
        ##for fpga_idx in range(self.num_fpgas):
        ##    for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
        ##        print(f"FPGA {fpga_idx} Kernel {i}")
        for fpga_idx in range(self.num_fpgas):
            for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
                kernel.run_kernel(self.packed_list_per_fpga[fpga_idx][i], sim_time)
    
    def wait_for_kernels(self, sim_time):
        for fpga_idx in range(self.num_fpgas):
            for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
                kernel.wait_for_kernel(sim_time)

