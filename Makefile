.PHONY: all validate codegen build run clean

RUST_BIN := ../target/release/titanmk
SCHEDULE := samples/linear.json
GEN_DIR := generated
NVCC ?= nvcc
NVCC_FLAGS := -O3 -std=c++17 -I kernels -I $(GEN_DIR)
# Set USE_CUBLAS=1 to link cuBLAS for matmul parity benchmarks.
ifdef USE_CUBLAS
NVCC_FLAGS += -DTITAN_USE_CUBLAS -lcublas
endif

all: run

validate:
	cd .. && cargo build --release
	$(RUST_BIN) validate $(SCHEDULE) --verbose

codegen: validate
	mkdir -p $(GEN_DIR)
	$(RUST_BIN) gen-cuda $(SCHEDULE) -o $(GEN_DIR)/generated_megakernel.cu --tensor-len 8

build: codegen
	$(NVCC) $(NVCC_FLAGS) src/main.cu -o $(GEN_DIR)/titan-megakernel-cuda-demo

run: build
	./$(GEN_DIR)/titan-megakernel-cuda-demo

clean:
	rm -rf generated/*.cu generated/*.inc generated/titan-megakernel-cuda-demo
