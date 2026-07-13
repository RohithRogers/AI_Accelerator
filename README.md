# TinyML Accelerator

An instruction-driven, INT8 TinyML accelerator for FPGA. Version 1 executes MLP inference with a deliberately non-pipelined RTL datapath.

Start with [the RTL implementation guide](docs/implementation.md). It is the coding order, module contract, and integration reference for the hardware work.

## Repository map

- `rtl/` — synthesizable SystemVerilog, arranged by subsystem.
- `tb/` — self-checking unit and integration testbenches.
- `sim/` — simulator scripts, build files, and waveforms.
- `sw/compiler/` — ISA assembler and its round-trip decoder test; this is the current encoding reference.
- `docs/` — concise design references.

## Version 1 scope

The accelerator supports `LOAD`, `STORE`, `DENSE`, `ACT`, `NOP`, and `END`. Dense supports fused None, ReLU, and ReLU6 activation. Softmax remains software-side. UART and FPGA-specific BRAM mapping come after simulation is working.

## First check

From the repository root, run the existing compiler check:

```powershell
python -m sw.compiler.test_roundtrip
```

Then implement the RTL in the order in [docs/implementation.md](docs/implementation.md).
