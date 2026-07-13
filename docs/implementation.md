# RTL implementation guide

This document is the coding reference for Version 1 RTL: a programmable, non-pipelined INT8 MLP inference accelerator. The Python assembler in `sw/compiler/compiler.py` defines the instruction bit encoding; do not independently reinterpret it in RTL.

## Scope and design rules

- Implement inference only: `LOAD`, `STORE`, `DENSE`, `ACT`, `NOP`, and `END`.
- All tensor, weight, and bias elements are signed 8-bit two's-complement bytes. Addresses and lengths are unsigned 16-bit element addresses.
- Execute exactly one instruction at a time. A `DENSE` is internally iterative, but no later instruction may begin until its `done` is asserted.
- Use synchronous active-low reset (`rst_n`) consistently unless the eventual board wrapper requires otherwise.
- Prefer simple registered request/response interfaces over combinational memory reads. It makes inferred FPGA RAM and testbenches agree.

## Architecture

```text
host preload / readback
          |
          v
 +------------------+     +-------------------+
 | instruction mem  | --> | fetch + decoder   | ---+
 +------------------+     +-------------------+    |
                                                     v
 +------------------+     +-------------------+  +------------------+
 | host data memory | <-> | memory controller |  | controller FSM   |
 +------------------+     +-------------------+  +------------------+
                               |      |                 |
                    +----------+      +-----------------+
                    v                                     v
             scratchpad / output                     dense_engine
                                                    (start / busy / done)
                                                      |       |      |
                                             input vectors  params  results
```

The controller owns program sequencing and high-level memory copies. The dense engine owns arithmetic and its loop counters. Memory blocks own storage only; they must not contain instruction-specific control.

## Common package: write this first

Create `rtl/tinyml_pkg.sv` with all widths, opcodes, activations, and decoded instruction fields. Make these parameters overridable at the top level:

| Parameter | Initial value | Meaning |
| --- | ---: | --- |
| `IMEM_DEPTH_WORDS` | 1024 | 32-bit instruction words |
| `SPAD_DEPTH` | 4096 | signed 8-bit activation elements |
| `PARAM_DEPTH` | 65536 | signed 8-bit elements per parameter bank |
| `OUT_DEPTH` | 4096 | signed 8-bit host-visible output elements |
| `SIMD_WIDTH` | 4 | products consumed by one inner-loop iteration |
| `ACC_WIDTH` | 32 | signed accumulation width |
| `REQUANT_SHIFT` | 0 | arithmetic right shift before INT8 saturation |
| `RELU6_MAX` | 127 | encoded ReLU6 ceiling until scale metadata is added |

Use enumerated 8-bit values: `NOP=8'h00`, `LOAD=8'h01`, `STORE=8'h02`, `DENSE=8'h03`, `ACT=8'h04`, `END=8'h05`; `ACT_NONE=0`, `ACT_RELU=1`, `ACT_RELU6=2`. Treat any other opcode or activation as an error and halt cleanly.

Define a packed `decoded_instr_t` containing `valid`, `opcode`, `flags`, `activation`, and every 16-bit operand field. Unused fields must decode to zero. This gives every later block one stable interface.

## ISA and decoder contract

Each program word is 32 bits. The header is `[31:24] opcode`, `[23:16] flags`, `[15:8] activation`, `[7:0] reserved`.

| Opcode | Total words | Operand extraction |
| --- | ---: | --- |
| `NOP` | 1 | none |
| `LOAD` | 3 | W1 `{mem_addr, sp_addr}`; W2 `{length, reserved}` |
| `STORE` | 3 | W1 `{sp_addr, mem_addr}`; W2 `{length, reserved}` |
| `DENSE` | 4 | W1 `{input_addr, weight_addr}`; W2 `{output_addr, bias_addr}`; W3 `{input_len, output_len}` |
| `ACT` | 2 | W1 `{addr, length}` |
| `END` | 1 | none |

`instruction_fetch` accepts `start`, reads the header at `pc`, determines total word count from only `header[31:24]`, reads the remaining operands, and raises a one-cycle `instr_valid` with the assembled words. It returns `next_pc = pc + total_words`; the controller commits that PC only after accepting a valid instruction. Reject an illegal opcode or a nonzero header reserved byte before execution.

`instruction_decoder` is combinational: it has no PC, no memory port, and no state. It accepts the fetched word bundle, extracts fields exactly as in `decode_stream()` in `sw/compiler/test_roundtrip.py`, and raises `decode_error` for invalid opcode, reserved bits, unsupported activation, or a required zero operand-reserved field that is nonzero.

## Storage and ownership

| Block | Element type | Read/write owner in V1 |
| --- | --- | --- |
| Instruction memory | 32-bit words | host preload before start; fetch read-only during run |
| Scratchpad | signed INT8 | `LOAD`, `ACT`, and dense-engine output/input |
| Weight memory | signed INT8 | host preload; dense engine read-only |
| Bias memory | signed INT8 | host preload; dense engine read-only |
| Output memory | signed INT8 | `STORE` write; host read after done |
| Host data memory | signed INT8 | simulation model / UART bridge |

Use distinct `weight_memory` and `bias_memory` modules (or two banks of a parameter-memory wrapper). This is intentional: the current compiler starts both `weight_addr` and `bias_addr` at zero, so a single shared physical array would overwrite one with the other. If the compiler later allocates one unified parameter address space, the banks can be merged behind the same engine ports.

For V1 simulation, expose preload tasks or synthesizable host write ports on instruction, weight, bias, and host-data memories. Keep UART out of the initial datapath tests.

## External interfaces to lock before coding

Top-level execution interface:

```systemverilog
input  logic clk, rst_n;
input  logic start;          // sampled only while idle
output logic busy;
output logic done;           // one clock pulse
output logic error;
output logic [7:0] error_code;
```

Compute command interface (`controller_fsm` to `dense_engine`):

```systemverilog
input  logic start;
input  logic [15:0] input_addr, weight_addr, output_addr, bias_addr;
input  logic [15:0] input_len, output_len;
input  logic [7:0]  activation;
output logic busy, done, error;
```

The controller holds all command fields stable from `start` until it sees `done` or `error`. The dense engine samples fields when `start && !busy`, asserts `busy` on the next cycle, pulses exactly one of `done`/`error` for one cycle, then returns idle. Define zero input or output length as a decode/execute error, not a silent no-op.

For each single-port byte memory, use an explicit request interface: `rd_en`, `rd_addr`, `rd_data`, `wr_en`, `wr_addr`, `wr_data`. Specify a one-cycle read latency and register `rd_data`. The engine schedules reads accordingly; do not rely on simulator-specific asynchronous array behavior.

## Controller FSM

Use one registered FSM. Suggested state sequence:

```text
IDLE -> FETCH -> DECODE -> DISPATCH
                         |-> LOAD_COPY  -> RETIRE -> FETCH
                         |-> STORE_COPY -> RETIRE -> FETCH
                         |-> DENSE_START -> DENSE_WAIT -> RETIRE -> FETCH
                         |-> ACT_LOOP   -> RETIRE -> FETCH
                         |-> NOP        -> RETIRE -> FETCH
                         `-> END        -> DONE -> IDLE
```

`FETCH` waits for a complete instruction. `DECODE` latches the decoded struct. `DISPATCH` checks bounds and selects the executor. `RETIRE` replaces the PC with the fetch-produced `next_pc`; it is the only state that advances PC. `DONE` pulses top-level `done`. Any decoder failure, executor failure, or bounds failure goes to `ERROR`, pulses/holds `error` until the next reset or fresh `start` policy you document in the top-level.

`LOAD_COPY` and `STORE_COPY` use a 16-bit `remaining` counter and increment source/destination addresses once per completed byte transfer. They must correctly handle length 1. No overlap semantics are required in V1.

## Dense engine microarchitecture

The dense result for output neuron `o` is:

```text
acc = sign_extend(bias[bias_addr + o])
for i = 0 .. input_len-1:
    acc += sign_extend(spad[input_addr + i]) *
           sign_extend(weight[weight_addr + o*input_len + i])
y = saturate_int8(activate(arithmetic_shift_right(acc, REQUANT_SHIFT)))
spad[output_addr + o] = y
```

Weights are row-major by output neuron: the first `input_len` bytes belong to output neuron 0. This matches the compiler's documented `input_len * output_len` allocation and must be stated in all tests.

Implement the engine as a sub-FSM with `out_idx`, `chunk_base`, `lanes_valid`, and `acc` registers:

```text
IDLE -> READ_BIAS -> INIT_ACC -> ISSUE_VECTOR_READ -> WAIT_VECTOR_READ
     -> MAC_ACCUMULATE -> (more chunks ? ISSUE_VECTOR_READ : POSTPROCESS)
     -> WRITE_RESULT -> (more outputs ? READ_BIAS : DONE)
```

For every chunk, calculate `lanes_valid = min(SIMD_WIDTH, input_len - chunk_base)`. Mask lanes beyond `lanes_valid`; this is mandatory when `input_len` is not divisible by `SIMD_WIDTH`. `vector_loader` forms scratchpad and weight addresses. `simd_mac_array` multiplies signed 8-bit values into signed 16-bit products; `adder_tree` sign-extends and sums them to `ACC_WIDTH`; `accumulator` adds that sum to the running signed accumulator. Keep each stage registered for clarity even though the architecture is non-pipelined: it means sequential steps within one instruction, not a fully combinational critical path.

`bias_adder` can be folded into `INIT_ACC` for V1. Keep it as a small module only if that helps unit testing. `writeback_unit` applies requantization, activation, saturation, and the scratchpad write.

## Numerical contract

The current assembler quantizes weights and biases independently but does not emit per-layer rescale values. Therefore the RTL cannot yet reproduce a physically scaled floating-point MLP solely from a `DENSE` instruction. Establish this deterministic V1 behavior:

1. Signed INT8 × signed INT8 products accumulate in signed 32-bit arithmetic.
2. Bias is signed INT8, sign-extended into the same accumulator domain.
3. `REQUANT_SHIFT` is an arithmetic right shift applied after the final accumulation, initially a top-level parameter set to zero.
4. ReLU clamps negative values to zero. ReLU6 clamps to `[0, RELU6_MAX]`; with the current no-scale ISA, default `RELU6_MAX=127`.
5. Saturate last to `[-128, 127]` and write the low signed INT8 result.

Before claiming ML accuracy, extend the compiler/ISA to deliver a per-layer output shift or fixed-point multiplier, a bias scale compatible with `input_scale * weight_scale`, and an encoded ReLU6 ceiling. The control and datapath interfaces above leave `flags` reserved for that extension.

## File-by-file coding plan

1. `tinyml_pkg.sv`: constants, structs, helper saturation function.
2. `regfile/register_file.sv`: eight 16-bit registers; synchronous write, combinational reads. Keep status bits (`busy`, `done`, `error`) controller-owned or explicitly map them to R7, but do not use R0–R6 as implicit instruction operands in V1.
3. `frontend/instruction_decoder.sv`: pure field extraction and errors; verify it before any FSM.
4. `frontend/program_counter.sv`, `instruction_memory.sv`, `instruction_fetch.sv`: fetch variable-length instructions and prove PC progression.
5. `frontend/controller_fsm.sv`: first support NOP/END, then dispatch stubs for other opcodes.
6. `memory/*.sv`: byte memories and copy engine; unit-test registered read timing and address bounds.
7. `compute/activation_unit.sv` and `requantizer.sv`: pure combinational numerical blocks.
8. `compute/vector_loader.sv`, `simd_mac_array.sv`, `adder_tree.sv`, `accumulator.sv`: unit-test signed arithmetic and tail chunks.
9. `compute/dense_engine.sv`: integrate the compute sub-FSM for a single DENSE command.
10. Add controller dispatch for DENSE, LOAD, STORE, and ACT; then wire `top/tinyml_accelerator_top.sv`.
11. Add `host_if/uart_interface.sv` only after direct-memory end-to-end simulation passes.

## Verification milestones

| Milestone | Testbench | Pass condition |
| --- | --- | --- |
| ISA decoder | `tb_instruction_decoder.sv` | Compiler words decode field-for-field like `decode_stream()` |
| Fetch/PC | `tb_instruction_fetch.sv` | Mixed-length program reaches END with exact word addresses |
| Control | `tb_controller_fsm.sv` | NOP/END produces one `done`; bad opcode produces `error` |
| Memories | `tb_memory.sv` | Registered reads and LOAD/STORE copy exact INT8 bytes |
| Math primitives | `tb_activation_unit.sv`, `tb_requantizer.sv`, `tb_simd_mac_array.sv` | negative values, saturation, signed products, and tail masks pass |
| Dense engine | `tb_dense_engine.sv` | 4→3 and non-multiple SIMD vectors match a Python integer golden model |
| Top level | `tb_top.sv` | compiler-generated 4→8→3 program copies final bytes to output memory and pulses `done` |

Each self-checking testbench should use assertions, a timeout, and a nonzero failure exit. Start every test with a reset, preload memories through the testbench, pulse `start` for one cycle, and check both data and handshake timing.

## Integration checklist

- Instruction memory is indexed in 32-bit words; every other memory is indexed in INT8 elements.
- All arithmetic signals are explicitly declared `signed`; cast before multiply and before compare/saturate.
- Bounds-check `base + length` and the dense weight range `weight_addr + input_len*output_len` using a widened intermediate, not 16-bit wraparound.
- `END` must not advance into another instruction or cause a spurious second `done`.
- `done` is one clock only; `busy` stays high from accepted start through retirement.
- Compare decoded RTL fields with `python -m sw.compiler.test_roundtrip` output before debugging execution.

## Deferred work

Do not block RTL bring-up on softmax, convolution, pooling, layer normalization, UART, AXI-Lite, or pipelining. Add those after the direct-preload, dense-only test suite is stable.
