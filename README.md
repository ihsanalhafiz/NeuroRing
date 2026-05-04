# NeuroRing: Scalable SNN Accelerator

## Repository overview
NeuroRing is a scalable spiking neural network (SNN) accelerator project for FPGA-based systems.

## Folder description
- `aurora_ipcore`: Aurora IP core sources for FPGA-to-FPGA communication over the QSFP port.
- `conf`: Build and configuration files used to generate the bitstream, including kernel definitions and connectivity.
- `hls`: High-Level Synthesis (HLS) source code for the accelerator kernels.
- `host_py`: Python host code used to control and interact with the FPGA kernels.

