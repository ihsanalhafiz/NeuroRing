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
import matplotlib.pyplot as plt


import struct
from collections import defaultdict
import math
from functools import reduce

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
        # Persistent BO and layout info
        self.synapseListHandle = None
        self.header_words = 120000 * 128  # recorder words (up to 120k timesteps * 128 words/tick)
        self.header_bytes = self.header_words * 4
        self.tail_words_capacity = max(0, self.neuron_total) * 10000
        self.tail_bytes_capacity = self.tail_words_capacity * 4
        self.bo_size = self.header_bytes + self.tail_bytes_capacity
        
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
        total_bo_size = self.header_bytes + self.tail_bytes_capacity
        self.synapseListHandle = pyxrt.bo(self.device, total_bo_size, pyxrt.bo.normal, self.kernel_axon_loader.group_id(0))
        print(f"Allocated BO of {total_bo_size} bytes (header {self.header_bytes}, tail {self.tail_bytes_capacity})")
        
    def run_kernel(self, synapse_list_data, simulation_time):
        # Clear only the recorder region needed for this run
        header_clear_words = min(simulation_time * 128, self.header_words)
        if header_clear_words > 0:
            zero_header = np.zeros(header_clear_words, dtype=np.uint32)
            self.synapseListHandle.write(zero_header, 0)
            self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, header_clear_words * 4, 0)
        
        # Upload only the synapse tail for this kernel
        tail_offset = self.header_bytes
        self.synapseListHandle.write(synapse_list_data, tail_offset)
        self.synapseListHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, synapse_list_data.nbytes, tail_offset)

        print(f"write done kernel {self.kernel_axon_loader}")
        ### run the kernel
        self.runAxonLoader = self.kernel_axon_loader(self.synapseListHandle, self.neuron_start, self.neuron_total, self.dcstim_start, self.dcstim_total, 
                                                     self.dcstim_amp, simulation_time, 1)
        print(f"Running kernel {self.kernel_axon_loader}")
        self.runNeuroRing = self.kernel_neuroring(simulation_time, self.threshold, self.membrane_potential, self.amount_of_cores, self.neuron_start, self.neuron_total)
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
        # --- Extract synapse_list and packed_list ---
        print("Extracting synapse information...")
        # Precompute DC amplitude per population and convert to IEEE-754 bits
        dc_amp = net_dict["K_ext"] * 0.3
        dc_amp_bits_per_pop = np.array(dc_amp, dtype='<f4').view('<u4').astype(np.uint32)

        # Build a union of all target populations once
        try:
            all_targets = reduce(lambda a, b: a + b, self.net.pops)
        except Exception:
            # Fallback: if reduce fails for some reason, just use the first pop
            all_targets = self.net.pops[0]
            for p in self.net.pops[1:]:
                all_targets = all_targets + p

        # Precompute V_m arrays and base global IDs per population
        pop_base_gid = []
        vm_bits_per_pop = []
        for pop in self.net.pops:
            base_gid = pop[0].global_id
            pop_base_gid.append(base_gid)
            vm_arr = np.asarray(pop.get('V_m'), dtype='<f4')
            vm_bits_per_pop.append(vm_arr.view('<u4').astype(np.uint32))
        pop_base_gid = np.array(pop_base_gid, dtype=np.int64)

        # Gather all connections in 8 calls (per source population), not 64
        all_sources_list = []
        all_targets_list = []
        all_weights_list = []
        all_delays10_list = []
        all_pop_index_list = []

        for i, source_pop in enumerate(self.net.pops):
            connections = nest.GetConnections(source=source_pop, target=all_targets)
            if len(connections) == 0:
                continue
            conn_info = nest.GetStatus(connections, ['source', 'target', 'weight', 'delay'])
            if not conn_info:
                continue
            # Unzip once, then vectorize
            srcs, tgts, wgts, dlys = zip(*conn_info)
            srcs = np.asarray(srcs, dtype=np.int64)
            tgts = np.asarray(tgts, dtype=np.int64)
            wgts = np.asarray(wgts, dtype=np.float32)
            dly10 = (np.asarray(dlys, dtype=np.float32) * 10.0).astype(np.int32)

            all_sources_list.append(srcs)
            all_targets_list.append(tgts)
            all_weights_list.append(wgts)
            all_delays10_list.append(dly10)
            all_pop_index_list.append(np.full(srcs.shape[0], i, dtype=np.int16))

        if len(all_sources_list) == 0:
            # No synapses
            self.synapse_list = []
            self.synapse_list_per_fpga = []
            self.packed_list_per_fpga = []
            self.kernel_neuron_ranges_per_fpga = []
            return

        sources_arr = np.concatenate(all_sources_list)
        targets_arr = np.concatenate(all_targets_list)
        weights_arr = np.concatenate(all_weights_list)
        delays10_arr = np.concatenate(all_delays10_list)
        popidx_arr = np.concatenate(all_pop_index_list)

        # Sort all connections by source for grouping
        order = np.argsort(sources_arr, kind='mergesort')
        sources_arr = sources_arr[order]
        targets_arr = targets_arr[order]
        weights_arr = weights_arr[order]
        delays10_arr = delays10_arr[order]
        popidx_arr = popidx_arr[order]

        # Optional: keep a lightweight synapse_list compatible with prior code (without repeated V_m/DC per synapse)
        # This keeps external expectations similar, without significant overhead
        self.synapse_list = list(zip(
            sources_arr.tolist(),
            targets_arr.tolist(),
            delays10_arr.tolist(),
            weights_arr.tolist(),
            popidx_arr.astype(int).tolist()
        ))

        # Group by source neuron
        unique_sources, first_indices, counts_per_source = np.unique(
            sources_arr, return_index=True, return_counts=True
        )

        # 3. Calculate CUs per FPGA
        num_cus_per_fpga = [
            self.num_compute_units // self.num_fpgas + (1 if i < self.num_compute_units % self.num_fpgas else 0)
            for i in range(self.num_fpgas)
        ]

        # 4. Split unique sources into groups for each CU, grouped by FPGA
        synapse_list_per_fpga = []  # maintained for compatibility, though not used for packing
        cu_idx = 0
        for fpga_idx, cu_count in enumerate(num_cus_per_fpga):
            fpga_cu_synapses = []
            for _ in range(cu_count):
                start = cu_idx * unique_sources.shape[0] // self.num_compute_units
                end = (cu_idx + 1) * unique_sources.shape[0] // self.num_compute_units
                # Compatibility placeholder: keep empty lists instead of materializing per-synapse Python rows
                fpga_cu_synapses.append([])
                cu_idx += 1
            synapse_list_per_fpga.append(fpga_cu_synapses)

        self.synapse_list_per_fpga = synapse_list_per_fpga
            
        # --- Create packed_list ---
        self.packed_list_per_fpga = []
        self.kernel_neuron_ranges_per_fpga = []
        block_size = 10000
        

        # Prepare fast lookups for header values per unique source
        # For each unique source, determine its population index and V_m/DC bits
        # popidx_arr contains population index per connection; for unique sources, use the first occurrence
        unique_first_popidx = popidx_arr[first_indices].astype(int)

        # Compute V_m bits per unique source
        # offset within population is (source_gid - base_gid[pop])
        src_offsets = unique_sources - pop_base_gid[unique_first_popidx]
        vm_bits_for_unique_src = np.array([
            vm_bits_per_pop[p_idx][off] if 0 <= off < vm_bits_per_pop[p_idx].shape[0] else np.uint32(0)
            for p_idx, off in zip(unique_first_popidx, src_offsets)
        ], dtype=np.uint32)
        dc_bits_for_unique_src = dc_amp_bits_per_pop[unique_first_popidx]

        # Build packed lists per CU using vectorized slices
        self.packed_list_per_fpga = []
        self.kernel_neuron_ranges_per_fpga = []

        cu_idx = 0
        for fpga_idx, cu_count in enumerate(num_cus_per_fpga):
            fpga_packed_lists = []
            fpga_neuron_ranges = []

            for _ in range(cu_count):
                start_u = cu_idx * unique_sources.shape[0] // self.num_compute_units
                end_u = (cu_idx + 1) * unique_sources.shape[0] // self.num_compute_units
                group_sources = unique_sources[start_u:end_u]
                group_first = first_indices[start_u:end_u]
                group_counts = counts_per_source[start_u:end_u]
                group_popidx = unique_first_popidx[start_u:end_u]
                group_vm_bits = vm_bits_for_unique_src[start_u:end_u]
                group_dc_bits = dc_bits_for_unique_src[start_u:end_u]

                if group_sources.shape[0] > 0:
                    neuron_start = int(group_sources.min())
                    neuron_total = int(group_sources.max() - neuron_start + 1)
                else:
                    neuron_start = -1
                    neuron_total = 0
                fpga_neuron_ranges.append((neuron_start, neuron_total))

                cu_packed = np.zeros(neuron_total * block_size, dtype=np.uint32)

                if neuron_total > 0 and neuron_start >= 0 and group_sources.shape[0] > 0:
                    # For each existing source in the group, fill header and synapses
                    for src_gid, first_idx_for_src, cnt, pidx, vm_bits, dc_bits in zip(
                        group_sources, group_first, group_counts, group_popidx, group_vm_bits, group_dc_bits
                    ):
                        block_start = int((src_gid - neuron_start) * block_size)
                        cu_packed[block_start] = int(cnt)
                        cu_packed[block_start + 1] = np.uint32(dc_bits)
                        cu_packed[block_start + 2] = np.uint32(vm_bits)

                        if cnt > 0:
                            s = int(first_idx_for_src)
                            e = s + int(cnt)
                            tgts = (targets_arr[s:e].astype(np.uint32) & np.uint32(0xFFFFFF))
                            dlys = (delays10_arr[s:e].astype(np.uint32) & np.uint32(0xFF))
                            w_bits = weights_arr[s:e].astype('<f4').view('<u4').astype(np.uint32)
                            
                            # sort by target (stable). If you also want delay as tie-breaker, see lexsort below.
                            order = np.argsort(tgts, kind='stable')
                            tgts  = tgts[order]
                            dlys  = dlys[order]
                            w_bits = w_bits[order]
                            
                            packed_td = (tgts << np.uint32(8)) | dlys

                            base = block_start + 16
                            idxs = base + 2 * np.arange(cnt, dtype=np.int64)
                            cu_packed[idxs] = packed_td
                            cu_packed[idxs + 1] = w_bits

                fpga_packed_lists.append(cu_packed)
                cu_idx += 1

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
                    membrane_potential=net_dict["neuron_params"]["E_L"],
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

    def get_spike_array(self, fpga_idx, kernel_idx, sim_time):
        """
        Fetch raw spike recorder words for a specific FPGA and kernel, reshaped as (sim_time, 128).

        Each timestep contains 128 uint32 words, totaling 4096 bits. Word 0 holds neurons 1..32,
        with bit 0 representing neuron 1, bit 31 representing neuron 32, and so on.
        """
        raw = self.kernels_per_fpga[fpga_idx][kernel_idx].get_spike_recorder_array(sim_time)
        return np.asarray(raw, dtype=np.uint32).reshape((sim_time, 128))

    def decode_spike_array(self, fpga_idx, kernel_idx, sim_time, return_global_ids=True):
        """
        Decode spike words into neuron IDs per timestep.

        - If return_global_ids is True, neuron IDs are offset by the kernel's neuron_start.
        - Otherwise, neuron IDs are local to the kernel (1-based within the 4096-bit window).

        Returns a list of length sim_time where each entry is a list of neuron IDs that spiked at that timestep.
        """
        words_per_timestep = self.get_spike_array(fpga_idx, kernel_idx, sim_time)

        # Retrieve neuron_start for proper global ID mapping when requested.
        try:
            neuron_start, neuron_total = self.kernel_neuron_ranges_per_fpga[fpga_idx][kernel_idx]
        except Exception:
            neuron_start, neuron_total = 1, None  # Fallback to 1-based if mapping unavailable

        spikes_by_timestep = []
        for t in range(sim_time):
            timestep_words = words_per_timestep[t]
            spiking_neurons = []
            for word_index in range(128):
                word = int(timestep_words[word_index])
                if word == 0:
                    continue
                # Iterate through set bits only
                bit_pos = 0
                while word:
                    # Isolate least significant set bit
                    lsb = word & -word
                    bit_pos = (lsb.bit_length() - 1)
                    local_neuron_id = word_index * 32 + bit_pos + 1  # 1-based within 4096
                    if return_global_ids:
                        global_neuron_id = neuron_start + local_neuron_id - 1
                        spiking_neurons.append(global_neuron_id)
                    else:
                        spiking_neurons.append(local_neuron_id)
                    # Clear the least significant set bit
                    word &= word - 1
            # Optionally constrain to known neuron_total if provided
            if neuron_total is not None and return_global_ids:
                max_global = neuron_start + max(neuron_total - 1, 0)
                spiking_neurons = [n for n in spiking_neurons if neuron_start <= n <= max_global]
            spikes_by_timestep.append(spiking_neurons)

        return spikes_by_timestep

    def get_spike_events(self, fpga_idx, kernel_idx, sim_time, return_global_ids=True):
        """
        Return (times, neuron_ids) suitable for raster plotting.

        times and neuron_ids are 1D numpy arrays of equal length.
        """
        spikes_by_timestep = self.decode_spike_array(fpga_idx, kernel_idx, sim_time, return_global_ids=return_global_ids)
        times_list = []
        neuron_ids_list = []
        for t, neuron_ids in enumerate(spikes_by_timestep):
            if not neuron_ids:
                continue
            times_list.extend([t] * len(neuron_ids))
            neuron_ids_list.extend(neuron_ids)
        if len(times_list) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        return np.asarray(times_list, dtype=np.int32), np.asarray(neuron_ids_list, dtype=np.int32)

    def plot_raster(self, fpga_idx, kernel_idx, sim_time, return_global_ids=True, ax=None, markersize=1.5, color='k'):
        """
        Plot a raster of spikes for a given FPGA and kernel over sim_time timesteps.

        - return_global_ids: plots global neuron IDs if True; otherwise local 1..4096 IDs.
        - ax: optional matplotlib Axes to plot on.
        - markersize, color: styling for the scatter points.
        Returns the matplotlib Axes used.
        """
        times, neuron_ids = self.get_spike_events(fpga_idx, kernel_idx, sim_time, return_global_ids=return_global_ids)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        if times.size > 0:
            ax.scatter(times, neuron_ids, s=markersize, c=color, marker='.', linewidths=0)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Neuron ID' + (' (global)' if return_global_ids else ' (local)'))
        ax.set_title(f'Raster: FPGA {fpga_idx}, Kernel {kernel_idx}, T={sim_time}')
        ax.set_xlim(-0.5, sim_time - 0.5)
        return ax

    def get_all_spike_arrays(self, sim_time):
        """
        Retrieve raw spike arrays for all FPGAs and kernels.

        Returns a nested list: result[fpga_idx][kernel_idx] -> np.ndarray of shape (sim_time, 128)
        """
        all_arrays = []
        for fpga_idx in range(self.num_fpgas):
            fpga_arrays = []
            for kernel_idx, _ in enumerate(self.kernels_per_fpga[fpga_idx]):
                fpga_arrays.append(self.get_spike_array(fpga_idx, kernel_idx, sim_time))
            all_arrays.append(fpga_arrays)
        return all_arrays

    def decode_all_spikes(self, sim_time, return_global_ids=True):
        """
        Decode spikes for all FPGAs and kernels.

        Returns a nested list: result[fpga_idx][kernel_idx] -> list(length=sim_time) of lists of neuron IDs per timestep.
        """
        all_decoded = []
        for fpga_idx in range(self.num_fpgas):
            fpga_decoded = []
            for kernel_idx, _ in enumerate(self.kernels_per_fpga[fpga_idx]):
                fpga_decoded.append(self.decode_spike_array(fpga_idx, kernel_idx, sim_time, return_global_ids=return_global_ids))
            all_decoded.append(fpga_decoded)
        return all_decoded

    def plot_raster_all(self, sim_time, return_global_ids=True, figsize=(14, 6), markersize=1.5, color='k'):
        """
        Plot rasters for all FPGAs and kernels in a grid.

        Grid shape: rows = num_fpgas, cols = max kernels per FPGA. Returns (fig, axes).
        """
        max_kernels = max((len(klist) for klist in self.kernels_per_fpga), default=0)
        if max_kernels == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title('No kernels to plot')
            return fig, ax
        fig, axes = plt.subplots(self.num_fpgas, max_kernels, figsize=figsize, squeeze=False)
        for fpga_idx in range(self.num_fpgas):
            for kernel_idx in range(max_kernels):
                ax = axes[fpga_idx][kernel_idx]
                if kernel_idx < len(self.kernels_per_fpga[fpga_idx]):
                    self.plot_raster(fpga_idx, kernel_idx, sim_time, return_global_ids=return_global_ids, ax=ax, markersize=markersize, color=color)
                else:
                    ax.axis('off')
        fig.tight_layout()
        return fig, axes

