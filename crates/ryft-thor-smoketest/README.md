# ryft-thor-smoketest

Small validation binary for `thor` that runs the same embedded MLIR program on CPU first and then on the CUDA 13 PJRT
plugin.

Build and run it with:

```bash
RYFT_XLA_SYS_ARCHIVE=/home/monday/ryft/ryft-xla-sys-linux-arm64-cpu.tar.gz \
PJRT_PLUGIN_CUDA_13_LIB=/home/monday/ryft/pjrt-plugin-linux-arm64-cuda-13.tar.gz \
cargo run -p ryft-thor-smoketest --features cuda-13
```
