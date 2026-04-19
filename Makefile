.PHONY: all hooks tests clean

all: hooks tests

hooks:
	cargo build --release

tests:
	mkdir -p tests/bin
	nvcc tests/test.cu -o tests/bin/test_app

clean:
	cargo clean
	rm -rf tests/bin
