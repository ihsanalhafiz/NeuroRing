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
from functools import reduce

from utils_binding import *   # provides .index and .bitstreamFile
import pyxrt

class NeuroRingKernel:
    def __init__(self, simulation_time, threshold, membrane_potential, amount_of_cores, neuron_start, neuron_total, dcstim_start, dcstim_total, dcstim_amp, core_id):
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
        # Persistent BO and layout info
        self.synapseListHandle = None
        self.header_words = 120000 * 64  # recorder words (up to 120k timesteps * 128 words/tick)
        self.header_bytes = self.header_words * 4
        self.tail_words_capacity = 2048 * 10000
        self.tail_bytes_capacity = self.tail_words_capacity * 4
        self.bo_size = self.header_bytes + self.tail_bytes_capacity
        self.core_id = core_id
        
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
        
        # Allocate persistent BO once per kernel and keep it for reuse across runs
        if self.tail_bytes_capacity < 0:
            self.tail_bytes_capacity = 0
        self.total_bo_size = self.header_bytes + self.tail_bytes_capacity
        self.synapseListHandle = pyxrt.bo(self.device, self.total_bo_size, pyxrt.bo.normal, self.kernel_axon_loader.group_id(0))
        print(f"Allocated BO of {self.total_bo_size} bytes (header {self.header_bytes}, tail {self.tail_bytes_capacity})")
        
    def run_kernel(self, synapse_list_data, simulation_time):
        header_clear_words = min(simulation_time * 128, self.header_words)
        if header_clear_words > 0:
            zero_header = np.zeros(header_clear_words, dtype=np.uint32)
            self.synapseListHandle.write(zero_header, 0)
            self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, header_clear_words * 4, 0)

        # Clear only the recorder region needed for this run
        tail_offset = self.header_bytes
        self.synapseListHandle.write(synapse_list_data, 0)
        self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, synapse_list_data.nbytes, 0)

        print(f"write done kernel {self.kernel_axon_loader}")
        ### run the kernel
        self.runAxonLoader = self.kernel_axon_loader(self.synapseListHandle, self.neuron_start, self.neuron_total, self.dcstim_start, self.dcstim_total, 
                                                     self.dcstim_amp, simulation_time, 1, self.core_id, self.amount_of_cores)
        print(f"Running kernel {self.kernel_axon_loader}")
        self.runNeuroRing = self.kernel_neuroring(simulation_time+1, self.threshold, self.membrane_potential, self.amount_of_cores, self.neuron_start, self.neuron_total, self.core_id)
        print(f"Running kernel {self.kernel_neuroring}")    


    def wait_for_kernel(self, sim_time):
        self.runAxonLoader.wait()
        print(f"Axon loader run complete {self.kernel_axon_loader}")
        self.runNeuroRing.wait()
        print(f"Kernel run complete {self.kernel_neuroring}")
        # Allow run handles to be GC'd between runs
        self.runAxonLoader = None
        self.runNeuroRing = None

    
    def wait_for_axon(self):
        self.runAxonLoader.wait()
        print(f"Axon loader run complete {self.kernel_axon_loader}")

    def wait_for_neuroring(self):
        self.runNeuroRing.wait()
        print(f"Kernel run complete {self.kernel_neuroring}")
        
    def get_spike_recorder_array(self, sim_time):
        # Read back 128 words per timestep (4096 bits)
        self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, (sim_time*128*4), 0)
        self.SpikeRecorder_array = np.frombuffer(self.synapseListHandle.read((sim_time*128*4), 0), dtype=np.uint32)
        return self.SpikeRecorder_array


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
        
        # Precompute DC amplitude per population and convert to IEEE-754 bits
        self.dc_amp = net_dict["K_ext"] * 0.3
        self.dc_amp_bits_per_pop = np.array(self.dc_amp, dtype='<f4').view('<u4').astype(np.uint32)

        # --- Extract synapse_list and packed_list ---
        print("Extracting synapse information...")
        
        # Get all synapse connections from NEST
        synapse_list = []
        for i, target_pop in enumerate(net.pops):
            for j, source_pop in enumerate(net.pops):
                if net.num_synapses[i][j] > 0:
                    connections = nest.GetConnections(source=source_pop, target=target_pop)
                    if len(connections) > 0:
                        conn_info = nest.GetStatus(connections, ['source', 'target', 'weight', 'delay'])
                        for conn in conn_info:
                            source = int(conn[0])      # source as integer
                            target = int(conn[1])      # target as integer
                            weight = float(conn[2])    # weight as float
                            delay = int(conn[3] * 10)  # delay * 10 as integer
                            source_pop_idx = j          # source population index
                            synapse_list.append([source, target, delay, weight, source_pop_idx])
        
        # Sort synapse data by source neuron
        synapse_list.sort(key=lambda x: x[0])
        self.synapse_list = synapse_list
        
        # Group synapses by source neuron
        source_dict = defaultdict(list)
        for synapse in synapse_list:
            source_dict[synapse[0]].append(synapse)
        
        # Sort synapses by destination within each source neuron
        for source_neuron in source_dict:
            source_dict[source_neuron].sort(key=lambda x: x[1])  # Sort by destination (index 1)
        
        # Get all neurons in the network (not just those with synapses)
        all_neurons = set()
        for pop in net.pops:
            for neuron in pop:
                all_neurons.add(neuron.global_id)
        
        # Sort all neurons
        sources = sorted(all_neurons)
        total_neurons = len(sources)
        
        print(f"Total neurons: {total_neurons}")
        print(f"Total synapses: {len(synapse_list)}")
        print(f"Neurons with synapses: {len(source_dict)}")
        
        # Calculate neurons per compute unit (max 2048 per CU)
        neurons_per_cu = 2048
        
        # Calculate compute units per FPGA
        cus_per_fpga = [max(1, num_compute_units // num_fpgas + (1 if i < num_compute_units % num_fpgas else 0)) 
                        for i in range(num_fpgas)]
        
        print(f"Debug: num_compute_units={num_compute_units}, num_fpgas={num_fpgas}")
        print(f"Debug: cus_per_fpga={cus_per_fpga}")
        
        # Create 3D structure: synapse_per_cu[fpga][cu][synapse_data]
        self.synapse_per_cu = []
        self.kernel_neuron_ranges_per_fpga = []
        
        # Initialize the full structure first
        for fpga_idx in range(num_fpgas):
            fpga_synapses = []
            fpga_neuron_ranges = []
            
            for cu_idx in range(cus_per_fpga[fpga_idx]):
                fpga_synapses.append([])
                fpga_neuron_ranges.append((0, 0))
            
            self.synapse_per_cu.append(fpga_synapses)
            self.kernel_neuron_ranges_per_fpga.append(fpga_neuron_ranges)
        
        # Now fill in the actual data
        neuron_idx = 0
        for fpga_idx in range(num_fpgas):
            for cu_idx in range(cus_per_fpga[fpga_idx]):
                if neuron_idx >= total_neurons:
                    break  # No more neurons to assign
                
                cu_neuron_start = sources[neuron_idx]  # Actual source neuron ID
                cu_neuron_count = min(neurons_per_cu, total_neurons - neuron_idx)
                
                # Create array indexed by neuron source (not by synapse)
                cu_synapses_by_neuron = []
                for i in range(cu_neuron_count):
                    if neuron_idx + i < len(sources):
                        source_neuron = sources[neuron_idx + i]
                        if source_neuron in source_dict:
                            # Apply circular sorting for this CU's synapse list
                            cu_synapses = source_dict[source_neuron].copy()
                            cu_synapses.sort(key=lambda x: self._circular_sort_key(x[1], cu_neuron_start, cu_neuron_start + cu_neuron_count - 1))
                            cu_synapses_by_neuron.append(cu_synapses)
                        else:
                            cu_synapses_by_neuron.append([])  # Empty list for neurons without synapses
                
                # Update the pre-allocated structure
                self.synapse_per_cu[fpga_idx][cu_idx] = cu_synapses_by_neuron
                self.kernel_neuron_ranges_per_fpga[fpga_idx][cu_idx] = (cu_neuron_start, cu_neuron_count)
                
                neuron_idx += cu_neuron_count
                
                if neuron_idx >= total_neurons:
                    break
            
            if neuron_idx >= total_neurons:
                break
        
        # Print distribution information
        print(f"Distribution across {num_fpgas} FPGAs:")
        for fpga_idx in range(num_fpgas):
            print(f"  FPGA {fpga_idx}: {len(self.synapse_per_cu[fpga_idx])} compute units")
            for cu_idx in range(len(self.synapse_per_cu[fpga_idx])):
                cu_synapses_by_neuron = self.synapse_per_cu[fpga_idx][cu_idx]
                neuron_start, neuron_count = self.kernel_neuron_ranges_per_fpga[fpga_idx][cu_idx]
                
                # Count total synapses in this compute unit
                total_synapses_in_cu = sum(len(neuron_synapses) for neuron_synapses in cu_synapses_by_neuron)
                
                print(f"    CU {cu_idx}: neurons {neuron_start}-{neuron_start + neuron_count - 1} ({neuron_count} neurons, {total_synapses_in_cu} synapses)")
        
        # Create pack_synapse_per_cu array for FPGA upload
        self.pack_synapse_per_cu = []
        for fpga_idx in range(num_fpgas):
            fpga_packed = []
            for cu_idx in range(cus_per_fpga[fpga_idx]):  # Use cus_per_fpga instead of len(self.synapse_per_cu)
                # Get neuron info for this CU (may be empty if no neurons assigned)
                if cu_idx < len(self.synapse_per_cu[fpga_idx]):
                    cu_synapses_by_neuron = self.synapse_per_cu[fpga_idx][cu_idx]
                    neuron_start, neuron_count = self.kernel_neuron_ranges_per_fpga[fpga_idx][cu_idx]
                else:
                    cu_synapses_by_neuron = []
                    neuron_start, neuron_count = 0, 0
                
                # Allocate: spike recorder area (120000*64) + neuron data (10000*2048)
                spike_recorder_size = 120000 * 64
                neuron_data_size = 10000 * 2048
                total_elements = spike_recorder_size + neuron_data_size
                cu_packed = np.zeros(total_elements, dtype=np.uint32)
                
                # Spike recorder area (first 120000*64 elements) is already initialized to 0
                # Neuron data starts at index 120000*64
                
                # Only fill parameter data if this CU actually has neurons
                if neuron_count > 0:
                    # Fill parameter data for each neuron (first 16 elements per neuron)
                    for neuron_idx in range(neuron_count):
                        if neuron_idx < len(cu_synapses_by_neuron):
                            neuron_synapses = cu_synapses_by_neuron[neuron_idx]
                            actual_neuron_id = neuron_start + neuron_idx
                            
                            # Calculate base index for this neuron in the packed array (after spike recorder area)
                            base_idx = spike_recorder_size + (neuron_idx * 10000)
                            
                            # Array[0]: total synapse count for current neuron
                            cu_packed[base_idx + 0] = len(neuron_synapses)
                            
                            # Array[1]: DC stimulus amplitude (need to find population for this neuron)
                            if neuron_synapses and len(neuron_synapses) > 0:
                                # Get population info from first synapse (assuming all synapses from same neuron have same pop)
                                source_pop_idx = neuron_synapses[0][4]  # population index from synapse data
                                if source_pop_idx < len(self.dc_amp_bits_per_pop):
                                    cu_packed[base_idx + 1] = self.dc_amp_bits_per_pop[source_pop_idx]
                            
                            # Array[2]: membrane potential initialization
                            # Use population index from synapse data for direct access
                            if neuron_synapses and len(neuron_synapses) > 0:
                                source_pop_idx = neuron_synapses[0][4]  # population index from synapse data
                                if source_pop_idx < len(net.pops):
                                    pop = net.pops[source_pop_idx]
                                    local_idx = actual_neuron_id - pop[0].global_id
                                    
                                    # Handle V_m as single value or array
                                    v_m_data = pop.get('V_m')
                                    if hasattr(v_m_data, '__len__'):  # If it's an array
                                        if local_idx < len(v_m_data):
                                            v_m_value = v_m_data[local_idx]
                                            # Convert to uint32 (IEEE-754 bits)
                                            v_m_bits = struct.unpack('<I', struct.pack('<f', v_m_value))[0]
                                            cu_packed[base_idx + 2] = v_m_bits
                                    else:  # If it's a single value
                                        v_m_value = v_m_data
                                        # Convert to uint32 (IEEE-754 bits)
                                        v_m_bits = struct.unpack('<I', struct.pack('<f', v_m_value))[0]
                                        cu_packed[base_idx + 2] = v_m_bits
                            
                            # Now add synapse data after the 16 parameter slots
                            # Each synapse takes 2 array slots: [destination+delay, weight]
                            for synapse_idx, synapse in enumerate(neuron_synapses):
                                if synapse_idx * 2 + 16 < 10000:  # Ensure we don't exceed neuron allocation
                                    synapse_base = base_idx + 16 + (synapse_idx * 2)
                                    
                                    # Slot 1: destination << 8 | delay
                                    destination = synapse[1]  # target neuron ID
                                    delay = synapse[2]        # delay value
                                    dest_delay = (destination << 8) | (delay & 0xFF)
                                    cu_packed[synapse_base + 0] = dest_delay
                                    
                                    # Slot 2: weight as IEEE-754 bits
                                    weight = synapse[3]  # weight value
                                    weight_bits = struct.unpack('<I', struct.pack('<f', weight))[0]
                                    cu_packed[synapse_base + 1] = weight_bits
                
                fpga_packed.append(cu_packed)
            self.pack_synapse_per_cu.append(fpga_packed)

    def _circular_sort_key(self, target, cu_start, cu_end):
        """
        Helper method for circular sorting of synapses within a CU.
        Targets within the CU range come first, then smaller targets wrap to the end.
        
        Args:
            target: Target neuron ID
            cu_start: Starting neuron ID for this CU
            cu_end: Ending neuron ID for this CU
        
        Returns:
            Sort key that implements circular ordering
        """
        if cu_start <= target <= cu_end:
            # Target is within CU range - comes first, sorted normally
            return target
        else:
            # Target is outside CU range - comes after, sorted normally but offset
            return target + 1000000  # Large offset to ensure it comes after CU range targets

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
                    membrane_potential=net_dict["neuron_params"]["E_L"],
                    amount_of_cores = self.num_compute_units,
                    neuron_start=neuron_start,
                    neuron_total=neuron_total,
                    dcstim_start=stim_dict["dc_start"],
                    dcstim_total=stim_dict["dc_dur"],
                    dcstim_amp=np.average(self.net.DC_amp),
                    core_id=kernel_id
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
                kernel.run_kernel(self.pack_synapse_per_cu[fpga_idx][i], sim_time)
    
    def wait_for_kernels(self, sim_time):
        for fpga_idx in range(self.num_fpgas):
            for i, kernel in enumerate(self.kernels_per_fpga[fpga_idx]):
                kernel.wait_for_kernel(sim_time)
                
    def save_array_to_csv(self, array, filename):
        np.savetxt(filename, array, fmt='%u', delimiter=',')

