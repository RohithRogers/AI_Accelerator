# Compute Engine — Module Interface Reference

> **Location**: `rtl/compute/`  
> **Companion testbenches**: `tb/tb_simd_mac_array.sv`, `tb/tb_vector_loader.sv`,
> `tb/tb_requantizer.sv`, `tb/tb_dense_engine.sv`  
> **Python golden reference**: `sw/compiler/test_dense_golden.py`

---

## Overview

The compute engine implements a pipelined INT8 dense (fully-connected) layer.
The top-level `dense_engine` orchestrates eight sub-modules that form one
multiply-accumulate datapath:

```
Input/Weight Memory
       │
  vector_loader   ← address generator, lane masker
       │
  simd_mac_array  ← W parallel signed 8×8 multipliers
       │
   adder_tree     ← W×16-bit → 32-bit signed sum
       │
  accumulator     ← running 32-bit accumulator register
       │
  bias_adder      ← add sign-extended INT8 bias
       │
  requantizer     ← arithmetic right-shift + sat8 → INT8
       │
 activation_unit  ← NONE / ReLU / ReLU6 / Sigmoid / Tanh
       │
 writeback_unit   ← registered write to scratchpad
```

All modules import `tinyml_pkg::*` for shared constants, enums, and the
`sat8()` saturation helper.

---

## Package Constants (`rtl/tinyml_pkg.sv`)

| Constant | Default | Description |
|---|---|---|
| `SIMD_WIDTH` | 8 | Parallel MAC lanes |
| `ACC_WIDTH` | 32 | Accumulator bit width |
| `REQUANT_SHIFT` | 0 | Arithmetic right-shift before saturation |
| `RELU6_MAX` | 127 | Ceiling for ReLU6 in INT8 scale |

> **Important**: These are package-level defaults. Each module declares them
> as overridable parameters. Pass them explicitly when instantiating
> sub-modules inside a wrapper or testbench.

---

## Module Interfaces

### `dense_engine` (top-level)

```systemverilog
module dense_engine #(
    parameter int SIMD_WIDTH     = 4,
    parameter int ACC_WIDTH      = 32,
    parameter int REQUANT_SHIFT  = 0,
    parameter int RELU6_MAX      = 127
) (
    input  logic        clk, rst_n,

    // ── Dispatch interface (driven by controller_fsm) ──────────────────────
    input  logic        start,          // Pulse HIGH for 1 cycle to begin
    input  logic [15:0] input_addr,     // Base address of activation vector in scratchpad
    input  logic [15:0] weight_addr,    // Base address of weight matrix row 0
    input  logic [15:0] output_addr,    // Base address of output vector in scratchpad
    input  logic [15:0] bias_addr,      // Base address of bias vector
    input  logic [15:0] input_len,      // Length of each input vector (K elements)
    input  logic [15:0] output_len,     // Number of output neurons to compute
    input  logic [7:0]  activation,     // ACT_NONE/ACT_RELU/ACT_RELU6/ACT_SIGMOID/ACT_TANH

    output logic        busy,           // HIGH while computing
    output logic        done,           // HIGH for exactly 1 cycle when complete
    output logic        error,          // HIGH if input_len or output_len = 0

    // ── Scratchpad read (W simultaneous ports) ─────────────────────────────
    output logic [15:0]       spad_rd_addr [SIMD_WIDTH],
    input  logic signed [7:0] spad_rd_data [SIMD_WIDTH],

    // ── Scratchpad write (1 port, registered) ──────────────────────────────
    output logic              spad_wr_en,
    output logic [15:0]       spad_wr_addr,
    output logic signed [7:0] spad_wr_data,

    // ── Weight RAM read (W simultaneous ports) ─────────────────────────────
    output logic [15:0]       weight_rd_addr [SIMD_WIDTH],
    input  logic signed [7:0] weight_rd_data [SIMD_WIDTH],

    // ── Bias RAM read (1 port) ─────────────────────────────────────────────
    output logic [15:0]       bias_rd_addr,
    input  logic signed [7:0] bias_rd_data
);
```

**Timing contract**:
- Assert `start` for exactly **1 clock cycle**.
- The engine becomes `busy` one cycle later. Do NOT re-assert `start` while `busy`.
- `done` pulses for **1 cycle** once every output neuron has been written back.
- All memory interfaces are **combinational reads** (0 wait states). If your
  RAM has a 1-cycle latency, add an extra `ST_WAIT_MEM`-style state (the FSM
  already includes one).

**Memory layout contract** (caller must preload before `start`):

| Resource | Address range | Content |
|---|---|---|
| Scratchpad | `input_addr .. input_addr+input_len-1` | Input activation INT8 |
| Weight RAM | `weight_addr .. weight_addr + output_len*input_len - 1` | Weight matrix row-major INT8 |
| Bias RAM | `bias_addr .. bias_addr + output_len - 1` | Bias vector INT8 |
| Scratchpad | `output_addr .. output_addr + output_len - 1` | **Written** by engine |

---

### `vector_loader`

```systemverilog
module vector_loader #(parameter int SIMD_WIDTH = 4) (
    input  logic        clk, rst_n,

    input  logic        start,             // 1-cycle pulse to begin new vector
    input  logic        next_chunk,        // 1-cycle pulse to advance to next chunk
    input  logic [15:0] base_sp_addr,      // Activation base address
    input  logic [15:0] base_weight_addr,  // Weight base address (current output neuron)
    input  logic [15:0] input_len,         // Total elements in the vector

    output logic        busy,              // HIGH while iterating chunks
    output logic        chunk_valid,       // HIGH when addresses are stable

    output logic [15:0] spad_rd_addr   [SIMD_WIDTH],
    output logic [15:0] weight_rd_addr [SIMD_WIDTH],

    output logic [SIMD_WIDTH-1:0] lanes_valid,  // 1 = lane contains real data
    output logic        last_chunk              // HIGH on final chunk
);
```

**How to use inside `dense_engine`**:

1. Assert `start` for 1 cycle; `busy` and `chunk_valid` go HIGH next cycle.
2. Read `spad_rd_addr` / `weight_rd_addr` to present to memories.
3. Wait 1 cycle for memory to return data (RAM latency).
4. Pass data to `simd_mac_array`, then assert `next_chunk` if `!last_chunk`.
5. When `last_chunk` is HIGH, do NOT assert `next_chunk`; proceed to post-process.

**Lane masking**: `lanes_valid[i] = 0` for tail elements when
`input_len` is not a multiple of `SIMD_WIDTH`. The MAC array zeroes those
products automatically.

---

### `simd_mac_array`

```systemverilog
module simd_mac_array #(parameter int SIMD_WIDTH = 4) (
    input  logic signed [7:0]  act_data   [SIMD_WIDTH],
    input  logic signed [7:0]  weight_data[SIMD_WIDTH],
    input  logic [SIMD_WIDTH-1:0] lanes_valid,

    output logic signed [15:0] products   [SIMD_WIDTH]
);
```

Pure **combinational**. No clock. Outputs are valid the same cycle inputs arrive.
Connect `lanes_valid` directly from `vector_loader` so tail elements produce `0`.

---

### `adder_tree`

```systemverilog
module adder_tree #(
    parameter int SIMD_WIDTH = 4,
    parameter int ACC_WIDTH  = 32
) (
    input  logic signed [15:0]      products[SIMD_WIDTH],
    output logic signed [ACC_WIDTH-1:0] sum_out
);
```

Pure **combinational**. Sign-extends each 16-bit product to `ACC_WIDTH` before
summing to prevent overflow. Connect directly between `simd_mac_array.products`
and `accumulator.sum_in`.

---

### `accumulator`

```systemverilog
module accumulator #(parameter int ACC_WIDTH = 32) (
    input  logic        clk, rst_n,
    input  logic        clear,       // 1-cycle: reset to 0 (start of new neuron)
    input  logic        accumulate,  // 1-cycle: add sum_in to register
    input  logic signed [ACC_WIDTH-1:0] sum_in,
    output logic signed [ACC_WIDTH-1:0] acc_out
);
```

`clear` and `accumulate` are mutually exclusive. Assert `clear` when beginning
a new output neuron. Assert `accumulate` each time `adder_tree` presents a
valid chunk sum.

---

### `bias_adder`

```systemverilog
module bias_adder #(parameter int ACC_WIDTH = 32) (
    input  logic signed [ACC_WIDTH-1:0] acc_in,
    input  logic signed [7:0]           bias_in,
    output logic signed [ACC_WIDTH-1:0] sum_out
);
```

Pure **combinational**. Sign-extends `bias_in` to `ACC_WIDTH` before adding.
Wire `acc_in` from `accumulator.acc_out` and `bias_in` from the bias RAM output.

---

### `requantizer`

```systemverilog
module requantizer #(
    parameter int ACC_WIDTH     = 32,
    parameter int REQUANT_SHIFT = 0
) (
    input  logic signed [ACC_WIDTH-1:0] acc_in,
    output logic signed [7:0]           val_out
);
```

Pure **combinational**. Performs `val_out = sat8(acc_in >>> REQUANT_SHIFT)`.
`REQUANT_SHIFT` is a **compile-time** parameter. If you need a runtime-variable
shift (V1.1 `shift` field), you must either:
- Re-parameterise the module and pass a runtime value, **or**
- Instantiate multiple requantizers with different shifts and mux the output.

> **V1.1 note**: The compiler emits a signed 8-bit `shift` field per
> instruction. To use it in RTL, replace `REQUANT_SHIFT` with a port
> `input logic signed [7:0] shift_n` and compute
> `acc_in >>> shift_n` in always_comb.

---

### `activation_unit`

```systemverilog
module activation_unit #(parameter int RELU6_MAX = 127) (
    input  logic [7:0]        act_type,  // Use ACT_* constants from tinyml_pkg
    input  logic signed [7:0] data_in,
    output logic signed [7:0] data_out
);
```

Pure **combinational**. Supported `act_type` values:

| Constant | Value | Behaviour |
|---|---|---|
| `ACT_NONE` | `8'h00` | Pass-through |
| `ACT_RELU` | `8'h01` | `max(0, x)` |
| `ACT_RELU6` | `8'h02` | `clamp(x, 0, RELU6_MAX)` |
| `ACT_SIGMOID` | `8'h03` | LUT (256 entries, input scale ≈ 0.0625) |
| `ACT_TANH` | `8'h04` | LUT (256 entries, input scale ≈ 0.0625) |

---

### `writeback_unit`

```systemverilog
module writeback_unit (
    input  logic              clk, rst_n,
    input  logic              write_enable,  // Assert for 1 cycle to commit
    input  logic [15:0]       write_addr,    // Scratchpad destination address
    input  logic signed [7:0] write_data,    // Activated output byte

    output logic              spad_wr_en,
    output logic [15:0]       spad_wr_addr,
    output logic signed [7:0] spad_wr_data
);
```

**Registered** — outputs `spad_wr_en/addr/data` appear **1 cycle after**
`write_enable` is asserted. Account for this extra cycle in any testbench that
reads `spad_mem[]` immediately after `done`.

---

## Dataflow Timing Diagram (1 output neuron, input_len = 8, SIMD_WIDTH = 4)

```
Cycle │  State          │ Key actions
──────┼─────────────────┼──────────────────────────────────────
  0   │  IDLE           │  start asserted
  1   │  READ_BIAS       │  bias_rd_addr issued
  2   │  INIT_ACC        │  acc_clear=1, vloader_start=1
  3   │  LOAD_CHUNK      │  chunk 0 addresses presented
  4   │  WAIT_MEM        │  memory returns data
  5   │  MAC (chunk 0)   │  acc_accumulate=1, vloader_next_chunk=1
  6   │  LOAD_CHUNK      │  chunk 1 addresses presented
  7   │  WAIT_MEM        │  memory returns data
  8   │  MAC (chunk 1)   │  acc_accumulate=1; last_chunk → POSTPROCESS
  9   │  POSTPROCESS     │  biased_sum / requant / activation computed
 10   │  WRITE_RESULT    │  wb_enable=1 → writeback_unit latches
 11   │  DONE            │  done=1 pulsed (if output_len==1)
 12   │  spad write      │  spad_wr_en=1 (1-cycle late from writeback_unit)
```

For `output_len > 1` the FSM loops: after `WRITE_RESULT` it goes back to
`READ_BIAS` with `out_idx` and `cur_weight_base` incremented.

---

## Running the Testbenches

### Prerequisites
- [Icarus Verilog](https://bleyer.org/icarus/) v11+ (`iverilog`, `vvp`)
- Python 3.10+

### Python golden reference

```powershell
# from workspace root
python -m sw.compiler.test_dense_golden
```

### SIMD MAC Array unit test

```powershell
iverilog -g2012 -o sim_mac.out `
    rtl/tinyml_pkg.sv `
    rtl/compute/simd_mac_array.sv `
    tb/tb_simd_mac_array.sv
vvp sim_mac.out
```

### Vector Loader unit test

```powershell
iverilog -g2012 -o sim_vloader.out `
    rtl/tinyml_pkg.sv `
    rtl/compute/vector_loader.sv `
    tb/tb_vector_loader.sv
vvp sim_vloader.out
```

### Requantizer unit test

```powershell
iverilog -g2012 -o sim_requant.out `
    rtl/tinyml_pkg.sv `
    rtl/compute/requantizer.sv `
    tb/tb_requantizer.sv
vvp sim_requant.out
```

### Dense Engine integration test

```powershell
iverilog -g2012 -o sim_dense.out `
    rtl/tinyml_pkg.sv `
    rtl/compute/vector_loader.sv `
    rtl/compute/simd_mac_array.sv `
    rtl/compute/adder_tree.sv `
    rtl/compute/accumulator.sv `
    rtl/compute/bias_adder.sv `
    rtl/compute/requantizer.sv `
    rtl/compute/activation_unit.sv `
    rtl/compute/writeback_unit.sv `
    rtl/compute/dense_engine.sv `
    tb/tb_dense_engine.sv
vvp sim_dense.out
```

Expected final lines:

```
=== ALL SIMD MAC ARRAY TESTS PASSED ===
=== ALL VECTOR LOADER TESTS PASSED ===
=== ALL REQUANTIZER TESTS PASSED ===
=== ALL DENSE ENGINE TESTS PASSED ===
```

---

## Common Integration Pitfalls

| Pitfall | Fix |
|---|---|
| Memory has 1-cycle read latency | Add a wait state between `LOAD_CHUNK` and `MAC` (already present as `ST_WAIT_MEM`) |
| `start` held high for > 1 cycle | Only pulse `start` for **exactly 1 cycle** while `!busy` |
| Reading scratchpad immediately after `done` | Wait **1 extra cycle** for `writeback_unit`'s registered write to commit |
| `REQUANT_SHIFT` is 0 but compiler emits nonzero `shift` | Upgrade `requantizer` to accept a runtime port `shift_n` |
| `output_len` neuron addresses overlap | Ensure `output_addr + output_len <= scratchpad_size` |
| `weight_addr` not advanced per neuron | `dense_engine` handles this internally via `cur_weight_base += input_len` |
