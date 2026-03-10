# /*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
# */

ECHO=@echo

.PHONY: help xclbin all clean

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make xclbin NEURON_NUM=<2816> CORE_PER_FPGA=<14> FREQ=<300> PLATFORM=<xilinx_u55c_gen3x16_xdma_3_202210_1>"
	$(ECHO) "      Command to build xclbin files for Alveo platform"

# PART setting: uncomment the lines matching your Alveo card, or override them by make variable
#PART := xcu200-fsgd2104-2-e
#PLATFORM ?= xilinx_u200_gen3x16_xdma_2_202110_1

#PART := xcu250-figd2104-2L-e
#PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1

PART := xcu55c-fsvh2892-2L-e
PLATFORM ?= xilinx_u55c_gen3x16_xdma_3_202210_1

#PART := xcu50-fsvh2104-2-e
#PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1

#PART := xcu280-fsvh2892-2L-e
#PLATFORM ?= xilinx_u280_gen3x16_xdma_1_202211_1

# TARGET: set the build target, only hw target is supported for designs including GT kernel
TARGET := hw
include ./utils.mk

##############################################
# Define size configuration
##############################################
NEURON_NUM := 4096
CORE_PER_FPGA := 10
FREQ := 300

TEMP_DIR := ./_x.$(TARGET).NUM_$(NEURON_NUM).CORE_$(CORE_PER_FPGA).FREQ_$(FREQ)
BUILD_DIR := ./_build_dir.$(TARGET).NUM_$(NEURON_NUM).CORE_$(CORE_PER_FPGA).FREQ_$(FREQ)
LINK_OUTPUT := $(BUILD_DIR)/neuroringcore.link.xclbin


################## IP resource generation 

./ip_generation/aurora_64b66b_0/aurora_64b66b_0.xci: ./aurora_ipcore/tcl/gen_aurora_ip.tcl
	mkdir -p ip_generation; rm -rf ip_generation/aurora_64b66b_0; vivado -mode batch -source $^ -tclargs $(PART)

./ip_generation/axis_data_fifo_0/axis_data_fifo_0.xci: ./aurora_ipcore/tcl/gen_fifo_ip.tcl
	mkdir -p ip_generation; rm -rf ip_generation/axis_data_fifo_0; vivado -mode batch -source $^ -tclargs $(PART)


################## hardware build 
COMMFLAGS := --platform $(PLATFORM) --target $(TARGET) --save-temps 
HLSCFLAGS := --compile $(COMMFLAGS) -I .
LINKFLAGS := --link --optimize 3 $(COMMFLAGS) --vivado.impl.jobs 16 --vivado.synth.jobs 16

FREQ_MHZ := --kernel_frequency $(FREQ)

RTL_SRC := ./aurora_ipcore/rtl/*.v
RTL_SRC += ./ip_generation/aurora_64b66b_0/aurora_64b66b_0.xci 
RTL_SRC += ./ip_generation/axis_data_fifo_0/axis_data_fifo_0.xci

XCLBIN_OBJ := krnl_aurora_test_$(TARGET).xclbin
NEURORING_XCLBIN_OBJ := krnl_neuroring_$(TARGET).xclbin

$(TEMP_DIR)/krnl_aurora.xo: $(RTL_SRC) ./aurora_ipcore/tcl/pack_kernel.tcl
	mkdir -p $(TEMP_DIR)
	rm -rf vivado_pack_krnl_aurora; mkdir vivado_pack_krnl_aurora; cd vivado_pack_krnl_aurora; vivado -mode batch -source ../aurora_ipcore/tcl/pack_kernel.tcl -tclargs $(PART) $(NEURON_NUM) $(CORE_PER_FPGA) $(FREQ)

$(TEMP_DIR)/krnl_neuroring.xo: ./hls/NeuroRing.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(HLSCFLAGS) $(FREQ_MHZ) --kernel NeuroRing --temp_dir $(TEMP_DIR) --output $@ $^ --hls.pre_tcl ./conf/compile_hls.tcl -D NEURON_NUM=$(NEURON_NUM)

$(TEMP_DIR)/krnl_neuroring_poisson.xo: ./hls/NeuroRing_Poisson.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(HLSCFLAGS) $(FREQ_MHZ) --kernel NeuroRing_Poisson --temp_dir $(TEMP_DIR) --output $@ $^ --hls.pre_tcl ./conf/compile_hls.tcl -D NEURON_NUM=$(NEURON_NUM)

$(TEMP_DIR)/krnl_synapserouter.xo: ./hls/SynapseRouter.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(HLSCFLAGS) $(FREQ_MHZ) --kernel SynapseRouter --temp_dir $(TEMP_DIR) --output $@ $^ --hls.pre_tcl ./conf/compile_hls.tcl -D NEURON_NUM=$(NEURON_NUM)

$(BUILD_DIR)/$(NEURORING_XCLBIN_OBJ): $(TEMP_DIR)/krnl_neuroring.xo $(TEMP_DIR)/krnl_neuroring_poisson.xo $(TEMP_DIR)/krnl_synapserouter.xo
	mkdir -p $(BUILD_DIR)
	v++ $(LINKFLAGS) $(FREQ_MHZ) --temp_dir $(TEMP_DIR) --config ./conf/NeuroRing_NUM_$(NEURON_NUM)_CORE_$(CORE_PER_FPGA).cfg --output $@  $(+)
	cp -rf $(TEMP_DIR)/reports $(BUILD_DIR)
	cp -rf $(TEMP_DIR)/logs $(BUILD_DIR)
	rm -rf .ipcache
	rm -rf .Xil
	rm -rf ip_generation
	rm -rf vivado_pack_krnl_aurora
	rm -f *.log *.jou

################## all flow
xclbin: $(BUILD_DIR)/$(NEURORING_XCLBIN_OBJ)

all: xclbin


################## clean up
clean:
	$(RM) -rf ip_generation vivado_pack_krnl_aurora
	$(RM) -rf *.xo *.xclbin *.xclbin.info *.xclbin.link_summary *.jou *.log *.xo.compile_summary _x
	$(RM) -rf *.dat *.pb xsim.dir *.xml *.ltx *.csv *.protoinst *.wdb *.wcfg

clean_log:
	$(RM) -rf *.log *.jou

