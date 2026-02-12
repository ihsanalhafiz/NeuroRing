#!/usr/bin/env python3
"""
Two-population NEST network (Exc & Inh, 10 each).
Bidirectional sparse connections (E->I, I->E) with 10% probability.
Random weights & delays per connection.
DC current to every neuron from t=0.
Records spikes and exports realized connections + DC amplitudes
in a Python dict you can later convert for FPGA.

Tested with NEST 3.8.
"""

import nest
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------
#  FPGA synapse-word packing helpers (from NeuroRing.h layout)
#    64 bits total:
#      [63:32]  float Weight raw bits
#      [31:26]  Delay (6 bits)
#      [25:0?]  Actually [23:0] DstID (24 bits)  -- remaining 2 bits unused in union's layout padding
#  In C: union { struct { Weight:32; Delay:6; DstID:24; }; } synapse_list_t
#  We'll pack into np.uint64 accordingly.
# ------------------------------------------------------------------
def pack_syn_word(dst_id: int, delay_steps: int, weight_f: float) -> np.uint64:
    """Pack into NeuroRing synapse_word_t layout."""
    # float32 -> raw bits
    w_bits = np.frombuffer(np.float32(weight_f).tobytes(), dtype=np.uint32)[0].astype(np.uint64)
    d_bits = (np.uint64(delay_steps) & np.uint64(0x3F)) << 24
    dst_bits = np.uint64(dst_id) & np.uint64(0xFFFFFF)
    return (w_bits << 32) | d_bits | dst_bits


def build_and_simulate(seed=42,
                       n_exc=10,
                       n_inh=10,
                       p_connect=0.10,
                       exc_weight_mu=1.0,  exc_weight_sigma=0.2,
                       inh_weight_mu=1.0,   inh_weight_sigma=0.2,  # sign applied negative
                       delay_min=1.0,       delay_max=5.0,
                       dc_mu=200.0,         dc_sigma=20.0,
                       sim_time=100.0,
                       export_npz=None):
    """
    Build the network, run, and optionally export realized connections & DC table.
    """
    # ---------------- Kernel setup ----------------
    nest.ResetKernel()
    nest.SetKernelStatus({"rng_seed": seed, "local_num_threads": 1})

    # ---------------- Populations -----------------
    exc = nest.Create("iaf_psc_alpha", n_exc)
    inh = nest.Create("iaf_psc_alpha", n_inh)

    # Convenience lists of GIDs (ints)
    exc_ids = exc.tolist()
    inh_ids = inh.tolist()
    all_ids = exc_ids + inh_ids

    # ---------------- Record spikes ----------------
    spk_rec = nest.Create("spike_recorder")
    nest.Connect(exc + inh, spk_rec)

    # ---------------- Connections ------------------
    # We let NEST sample the weights/delays internally using nest.random distributions,
    # and we explicitly pass the NodeCollections.  No Python loops; avoids hashing errors.
    # Excitatory synapses positive
    w_exc = nest.random.normal(mean=exc_weight_mu, std=exc_weight_sigma)
    # Inhibitory synapses negative (sample positive then multiply -1)
    w_inh = nest.random.normal(mean=inh_weight_mu, std=inh_weight_sigma) * -1.0
    # Delay distribution (same used for both projections)
    d_rng  = nest.random.uniform(min=delay_min, max=delay_max)

    # E -> I
    nest.Connect(exc, inh,
                 conn_spec={'rule': 'pairwise_bernoulli', 'p': p_connect},
                 syn_spec={'weight': w_exc, 'delay': d_rng})

    # I -> E
    nest.Connect(inh, exc,
                 conn_spec={'rule': 'pairwise_bernoulli', 'p': p_connect},
                 syn_spec={'weight': w_inh, 'delay': d_rng})

    # ---------------- DC input ---------------------
    # One dc_generator per neuron with amplitude ~ N(dc_mu, dc_sigma)
    rng = np.random.default_rng(seed + 1000)  # decouple from NEST RNG
    dc_amps = []
    for gid in all_ids:
        amp = float(rng.normal(dc_mu, dc_sigma))
        dc = nest.Create("dc_generator", params={'amplitude': amp})
        nest.Connect(dc, [gid])
        dc_amps.append(amp)

    # ---------------- Simulate ---------------------
    print(f"Simulating {sim_time} ms …")
    nest.Simulate(sim_time)
    print("Done.")

    # ---------------- Spike counts -----------------
    evts = nest.GetStatus(spk_rec, "events")[0]
    senders = evts["senders"]
    unique, counts = np.unique(senders, return_counts=True)
    spike_counts = dict(zip(unique, counts))
    print("\nSpike counts per neuron:")
    for nid in all_ids:
        print(f"  Neuron {nid:3d}: {spike_counts.get(nid, 0)} spikes")

    # ---------------- Export realized connections -----------------
    # Grab actual synapses that NEST created so we can drive FPGA inputs.
    # E->I
    conns_EI = nest.GetConnections(source=exc, target=inh)
    st_EI = nest.GetStatus(conns_EI, keys=['source', 'target', 'weight', 'delay'])
    # I->E
    conns_IE = nest.GetConnections(source=inh, target=exc)
    st_IE = nest.GetStatus(conns_IE, keys=['source', 'target', 'weight', 'delay'])

    # Flatten to numpy arrays
    def status_to_arrays(st_list):
        srcs = np.array([d['source'] for d in st_list], dtype=np.int64)
        tgts = np.array([d['target'] for d in st_list], dtype=np.int64)
        wgts = np.array([d['weight'] for d in st_list], dtype=np.float32)
        dlys = np.array([d['delay']  for d in st_list], dtype=np.float32)
        return srcs, tgts, wgts, dlys

    src_EI, tgt_EI, w_EI, d_EI = status_to_arrays(st_EI)
    src_IE, tgt_IE, w_IE, d_IE = status_to_arrays(st_IE)

    # ---------------- FPGA packing prep -----------------
    # Map global NEST GIDs to contiguous global neuron IDs 0..N-1.
    # We'll assume Exc occupy 0..n_exc-1 and Inh occupy n_exc..n_exc+n_inh-1.
    gid_to_global = {gid: i for i, gid in enumerate(all_ids)}
    global_count = len(all_ids)

    # Delay quantization: NEST gives ms; NeuroRing Delay field = 6 bits.
    # Decide on "1 delay step = 1 ms" (change if you need different dt).
    def quantize_delay_ms(d_ms):
        d = int(round(d_ms))
        if d < 0: d = 0
        if d > 63: d = 63
        return d

    # Build per-source-neuron synapse lists (Python dict)
    syn_dict = {i: [] for i in range(global_count)}

    # E->I
    for s, t, w, d in zip(src_EI, tgt_EI, w_EI, d_EI):
        s_idx = gid_to_global[int(s)]
        t_idx = gid_to_global[int(t)]
        syn_dict[s_idx].append((t_idx, quantize_delay_ms(d), float(w)))

    # I->E
    for s, t, w, d in zip(src_IE, tgt_IE, w_IE, d_IE):
        s_idx = gid_to_global[int(s)]
        t_idx = gid_to_global[int(t)]
        syn_dict[s_idx].append((t_idx, quantize_delay_ms(d), float(w)))

    # ---------------- Optional NPZ export -----------------
    if export_npz is not None:
        export_path = Path(export_npz).expanduser().resolve()
        # Convert ragged syn_dict to packed arrays later; here save object npz
        np.savez_compressed(export_path,
                            seed=seed,
                            n_exc=n_exc,
                            n_inh=n_inh,
                            all_ids=np.array(all_ids, dtype=np.int64),
                            dc_amps=np.array(dc_amps, dtype=np.float32),
                            syn_dict=np.array(object, dtype=object))  # placeholder; see below
        print(f"Exported network metadata to {export_path}")

    # Return structures to caller (host integration step)
    return {
        "exc": exc,
        "inh": inh,
        "spike_recorder": spk_rec,
        "events": evts,
        "src_EI": src_EI, "tgt_EI": tgt_EI, "w_EI": w_EI, "d_EI": d_EI,
        "src_IE": src_IE, "tgt_IE": tgt_IE, "w_IE": w_IE, "d_IE": d_IE,
        "gid_to_global": gid_to_global,
        "syn_dict": syn_dict,
        "dc_amps": dc_amps,
    }


if __name__ == "__main__":
    build_and_simulate()
