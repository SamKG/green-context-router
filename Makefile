.PHONY: all hooks tests clean

all: hooks tests

hooks:
	cargo build --release
	cp target/release/libgreen_ctx_router.so target/release/libcuda.so
	cp target/release/libgreen_ctx_router.so target/release/libcuda.so.1

tests:
	mkdir -p tests/bin
	nvcc tests/test.cu -o tests/bin/test_app

clean:
	cargo clean
	rm -rf tests/bin
