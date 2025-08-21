# SynapseRouter Kernel

## Overview
The `SynapseRouter` is a free-running HLS kernel that routes synapse data from AxonLoader to 4 destination slots based on destination ranges. It acts as a sorting buffer that accumulates synapse data and outputs complete packets when slots are full.

## Functionality

### Input
- **SynapseStream**: 512-bit stream containing 8 synapse entries (64 bits each)
- Each synapse entry contains:
  - 24-bit destination (dst)
  - 8-bit delay
  - 32-bit weight

### Routing Logic
The kernel routes synapse data to 4 slots based on destination ranges:
- **Slot 0**: destinations 1-2048
- **Slot 1**: destinations 2049-4096
- **Slot 2**: destinations 4097-6144
- **Slot 3**: destinations 6145-8192

### Slot Buffers
- Each slot can hold up to 8 synapse entries (512 bits total)
- When a slot is full, it outputs the data and resets
- Initial buffer values are 0

### Output
- **SynapseOut**: 512-bit stream with 8-bit ID and 8-bit DEST
- **ID**: corresponds to slot number (0, 1, 2, 3)
- **DEST**: corresponds to slot number (0, 1, 2, 3)
- **Data**: 512 bits of synapse data from the full slot

### Special Handling
- **Last Flag Detection**: When input packet has delay=0xFE, it triggers output of all non-empty slots with last=1
- **Sequential Output**: If multiple slots are full, they output sequentially
- **Buffer Reset**: After output, slot buffers and counters are reset to 0

## Interface
```cpp
extern "C" void SynapseRouter(
    hls::stream<stream512u_t>    &SynapseStream,  // Input stream
    hls::stream<stream512u_t>    &SynapseOut);    // Output stream
```

## HLS Directives
- `#pragma HLS INTERFACE ap_ctrl_none port=return` - Free-running kernel
- `#pragma HLS PIPELINE II=1` - Pipeline with II=1 for high throughput
- `#pragma HLS ARRAY_PARTITION variable=slot_buffer complete dim=1` - Complete partitioning for parallel access
- `#pragma HLS UNROLL` - Unroll loops for parallel processing

## Usage Example
```cpp
// In your top-level design
hls::stream<stream512u_t> synapse_stream;
hls::stream<stream512u_t> synapse_out;

// Connect AxonLoader output to SynapseRouter input
AxonLoader(..., synapse_stream, ...);

// Route synapses to output
SynapseRouter(synapse_stream, synapse_out);

// Use synapse_out in downstream processing
```

## Data Flow
1. **Input Processing**: Kernel reads 512-bit packets from SynapseStream
2. **Destination Analysis**: Each of 8 synapse entries is analyzed for destination range
3. **Slot Assignment**: Synapse data is packed into appropriate slot buffer
4. **Output Generation**: When slots are full or last flag detected, data is output
5. **Buffer Management**: Output slots are reset for next cycle

## Performance Characteristics
- **Latency**: Minimal (pipelined operation)
- **Throughput**: 1 packet per cycle when processing
- **Resource Usage**: 4 × 512-bit buffers + counters
- **Clock Frequency**: Optimized for high-frequency operation

## Testing
Use `test_synapse_router.cpp` to verify functionality:
- Creates test packets with various destination ranges
- Simulates slot filling and output generation
- Tests last flag handling
