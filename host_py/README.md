## File description

The `host_py` directory contains the Python host-side code used to communicate with and control the FPGA accelerator.

| File | Description |
| --- | --- |
| `helpers.py` | Utility functions for deriving microcircuit parameters, converting PSP values to currents, loading spike recorder output, computing firing-rate and correlation metrics, and writing plots/JSON metadata. |
| `host_4096_5core_1fpga_quarter.py` | End-to-end host script for the quarter-scale 4096-neuron-per-core, 5-core, 1-FPGA run; builds the NEST network, initializes the NeuroRing bitstream, uploads synapse buffers, launches kernels, and records timing/power measurements. |
| `nest_network.py` | Small two-population NEST test network used to generate example excitatory/inhibitory connectivity, DC inputs, spike recordings, and packed synapse-word data for FPGA integration experiments. |
| `network.py` | Main PyNEST microcircuit `Network` class that derives scaled parameters, configures NEST, creates neuron populations and stimulation/recording devices, connects the cortical network, runs simulations, and evaluates spike output. |
| `network_params.py` | Default cortical microcircuit network parameters, including population sizes, connection probabilities, neuron model parameters, synaptic weights, delays, external input settings, and derived PSP/delay matrices. |
| `neuroring.py` | NeuroRing FPGA host interface built on `pyxrt`; prepares NEST synapse data for FPGA buffers, partitions neurons across compute units/FPGAs, initializes kernels, runs NeuroRing and SynapseRouter kernels, reads spike recorder buffers, and plots decoded FPGA spikes. |
| `sim_params.py` | Default simulation settings such as run duration, time resolution, random seed, threading, recording devices, output path, overwrite policy, progress printing, and metadata storage. |
| `stimulus_params.py` | Default optional stimulus settings for thalamic spike input and transient DC stimulation, including timing, rates, amplitudes, and target connection probabilities. |
| `utils_binding.py` | Xilinx/XRT command-line option helpers for selecting bitstreams, devices, compute units, platform JSON files, verbosity, and parsing platform memory-interface counts. |
