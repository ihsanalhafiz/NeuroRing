import os
import sys
import pyxrt
import contextlib
from neuroring import NeuroRingHost, NeuroRingKernel

from utils_binding import Options  # Ensure this module provides .index and .bitstreamFile


def runKernel(opt):
    # Open device and load xclbin
    device = pyxrt.device(opt.index)
    xclbin = pyxrt.xclbin(opt.bitstreamFile)
    uuid = device.load_xclbin(xclbin)
    
    print(f"Loaded XCLBIN UUID: {uuid.to_string()}")
    print(f"Device UUID from device object: {device.get_xclbin_uuid().to_string()}")
    print(f"XSA Name: {xclbin.get_xsa_name()}")
    print("")

    # Kernels Info
    print("KERNELS INFO:")
    kernels = xclbin.get_kernels()
    for k in kernels:
        print(f"  Kernel Name: {k.get_name()}")
        print(f"    Number of Arguments: {k.get_num_args()}")
        for i in range(k.get_num_args()):
            try:
                group_id = k.group_id(i)
                print(f"    Arg[{i}] Group ID: {group_id}")
            except Exception:
                print(f"    Arg[{i}] Group ID: Not applicable")
    print("")

    # Memory Banks Info
    print("MEMORY BANKS INFO:")
    mems = xclbin.get_mems()
    for mem in mems:
        if not mem.get_used():
            continue
        print(f"  Memory Tag: {mem.get_tag()}")
        print(f"    Index: {mem.get_index()}")
        print(f"    Base Address: 0x{mem.get_base_address():X}")
        print(f"    Size (KB): {mem.get_size_kb()}")
    print("")

    # Device info (optional additional info)
    print("DEVICE INFO:")
    for info_key in dir(pyxrt.xrt_info_device):
        if not info_key.startswith("__"):
            try:
                info_value = device.get_info(getattr(pyxrt.xrt_info_device, info_key))
                print(f"  {info_key}: {info_value}")
            except Exception:
                continue
    print("")


def main(args):
    opt = Options()
    Options.getOptions(opt, args)

    try:
        runKernel(opt)
        print("TEST PASSED")
        return 0

    except OSError as o:
        print(o)
        print("TEST FAILED")
        return -o.errno

    except AssertionError as a:
        print(a)
        print("TEST FAILED")
        return -1
    except Exception as e:
        print(e)
        print("TEST FAILED")
        return -1


if __name__ == "__main__":
    os.environ["Runtime.xrt_bo"] = "false"
    with open("xclbin_info_log.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            result = main(sys.argv)
    sys.exit(result)
