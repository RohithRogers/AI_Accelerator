# Dense-only FPGA implementation and UART loading plan

## Scope and baseline

Implement the accelerator as a dense/MLP engine only.  Do **not** include the
convolution files or the CONV controller paths.  Commit `7b7fce3`
(`Accelerator ready for NN ... still conv needs to be added`) is the intended
baseline: the latest commit says that CONV simulation is not working.

The compiler and RTL must use the same ISA version.  In particular, current
`imem.hex` uses a five-word `DENSE` instruction: header, addresses, lengths,
and a 32-bit Q31 requantization multiplier.  Do not use it with an older
four-word-DENSE decoder.

## RTL files for the FPGA project

Add these synthesizable sources, in dependency order, to Vivado/Quartus.

```text
rtl/tinyml_pkg.sv

rtl/frontend/program_counter.sv
rtl/frontend/instruction_memory.sv
rtl/frontend/instruction_fetch.sv
rtl/frontend/instruction_decoder.sv
rtl/frontend/controller_fsm.sv

rtl/memory/memory_controller.sv
rtl/memory/scratchpad_memory.sv
rtl/memory/weight_memory.sv
rtl/memory/bias_memory.sv
rtl/memory/host_data_memory.sv
rtl/memory/output_memory.sv

rtl/compute/activation_unit.sv
rtl/compute/accumulator.sv
rtl/compute/adder_tree.sv
rtl/compute/bias_adder.sv
rtl/compute/vector_loader.sv
rtl/compute/simd_mac_array.sv
rtl/compute/requantizer_runtime.sv
rtl/compute/writeback_unit.sv
rtl/compute/dense_engine.sv

rtl/host_if/uart_rx.sv
rtl/host_if/uart_tx.sv
rtl/host_if/uart_interface.sv

rtl/top/tinyml_accelerator_top.sv
rtl/top/tinyml_fpga_top.sv       # new board wrapper; synthesis top
```

Exclude `rtl/compute/conv/`, plus all `OP_CONV` / `OP_CONV_CFG` logic from the
top, controller, package, and compiler for this dense-only build.  Keep the
testbenches and Python files out of synthesis; they are verification tools.

## FPGA board wrapper

Create `tinyml_fpga_top` to connect the serial pins to the existing host
interface.  Its minimum external ports should be:

```systemverilog
module tinyml_fpga_top #(
  parameter int unsigned CLK_FREQ  = 50_000_000,
  parameter int unsigned BAUD_RATE = 115200
) (
  input  logic clk_50mhz,  // rename to the board oscillator frequency
  input  logic rst_n,      // external active-low reset, or generated internally
  input  logic uart_rx,
  output logic uart_tx,
  output logic led_busy,   // optional but strongly recommended
  output logic led_done,   // optional; latch/extend the one-cycle done pulse
  output logic led_error   // optional
);
```

Inside the wrapper:

1. Generate a clean internal `clk` and reset.  If the board button is
   active-high, invert it before connecting it to the core `rst_n`; preferably
   debounce and synchronize it.
2. Instantiate `uart_interface` with the *actual* clock frequency and desired
   baud rate.
3. Instantiate `tinyml_accelerator_top`.
4. Wire `start_pulse`, `busy`, `done`, `error`, `error_code`, and the complete
   host memory bus between those two blocks.
5. Assign `uart_rx` to UART receiver `rx` and UART transmitter `tx` to
   `uart_tx`.
6. Drive `led_busy = busy`, `led_error = error`; stretch or latch `done` for
   `led_done`, since `done` is only one clock cycle.

These signals are internal only and **do not belong in the constraint file**:
`host_wr_en`, `host_rd_en`, `host_target_id`, `host_addr`,
`host_wr_data32`, `host_wr_data8`, `host_rd_data32`, `host_rd_data8`,
`start_pulse`, `busy`, `done`, and `error`.

## Constraint-file signals

Only physical wrapper ports receive pin constraints.  The pin names below are
placeholders: replace `PIN_CLOCK`, `PIN_RESET`, `PIN_UART_RX`, `PIN_UART_TX`,
and LED pin names with the exact pins from the board schematic/manual.  Do not
copy placeholder pin names into a build.

### Xilinx Vivado XDC template

```tcl
# Clock oscillator: example is 50 MHz => 20 ns period.
set_property PACKAGE_PIN PIN_CLOCK [get_ports clk_50mhz]
set_property IOSTANDARD LVCMOS33 [get_ports clk_50mhz]
create_clock -name sys_clk -period 20.000 [get_ports clk_50mhz]

# Reset input.  Set PULLUP only if the board reset circuit requires it.
set_property PACKAGE_PIN PIN_RESET [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]
# set_property PULLUP true [get_ports rst_n]

# UART is standard 8N1 CMOS-level serial, not RS-232 voltage levels.
set_property PACKAGE_PIN PIN_UART_RX [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]
set_property PULLUP true [get_ports uart_rx]

set_property PACKAGE_PIN PIN_UART_TX [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
set_property DRIVE 8 [get_ports uart_tx]
set_property SLEW SLOW [get_ports uart_tx]

# Optional status LEDs
set_property PACKAGE_PIN PIN_LED_BUSY  [get_ports led_busy]
set_property PACKAGE_PIN PIN_LED_DONE  [get_ports led_done]
set_property PACKAGE_PIN PIN_LED_ERROR [get_ports led_error]
set_property IOSTANDARD LVCMOS33 [get_ports {led_busy led_done led_error}]
```

If your oscillator is 100 MHz, use `create_clock -period 10.000` and set
`CLK_FREQ = 100_000_000`; otherwise the UART baud divider will be wrong.

### Intel Quartus QSF/SDC template

```tcl
# .qsf
set_location_assignment PIN_CLOCK   -to clk_50mhz
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk_50mhz
set_location_assignment PIN_RESET   -to rst_n
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to rst_n
set_location_assignment PIN_UART_RX -to uart_rx
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to uart_rx
set_location_assignment PIN_UART_TX -to uart_tx
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to uart_tx
set_location_assignment PIN_LED_BUSY  -to led_busy
set_location_assignment PIN_LED_DONE  -to led_done
set_location_assignment PIN_LED_ERROR -to led_error

# .sdc
create_clock -name sys_clk -period 20.000 [get_ports {clk_50mhz}]
```

Use the board's required I/O bank voltage/standard instead of assuming
`LVCMOS33` / `3.3-V LVTTL`.  A USB-UART bridge normally already presents
3.3-V CMOS UART.  Never connect a true +/-12-V RS-232 port directly to FPGA
pins; use an RS-232 level shifter.

## Hex artifacts and memory map

The flow produces the following files in `flow/`:

| File | UART target | Physical memory | Address unit |
|---|---:|---|---|
| `imem.hex` | 0 | instruction memory | 32-bit word |
| `input.hex` | 1 | host-data memory | byte |
| — | 2 | scratchpad | byte |
| `weight.hex` | 3 | weight memory | byte |
| `bias.hex` | 4 | bias memory | byte |
| `golden_output.hex` | — | expected readback | byte |
| output readback | 5 | output memory | byte |

Weights, biases, inputs, and outputs are signed INT8 two's-complement bytes.
The textual hex files must be converted to raw bytes by the PC program before
they are sent over UART; do not transmit the ASCII characters `f` and `e` for
the value `0xfe`.

## UART protocol and placement

Use 8N1 UART (one start bit, eight data bits, no parity, one stop bit), LSB
first, initially at 115200 baud.  Existing command packets are:

```text
WRITE_MEM (0x01): [01] [target] [addr_hi] [addr_lo] [len_hi] [len_lo] [data...]
READ_MEM  (0x02): [02] [target] [addr_hi] [addr_lo] [len_hi] [len_lo]
START     (0x03): [03]
STATUS    (0x04): [04] -> [status_flags] [error_code]
```

For byte memories, `address` and `length` are bytes.  For IMEM, `address` is
a **word address**, while `length` is still the number of payload bytes and
must be divisible by four.  Pack each instruction word big-endian.  For
example, an `imem.hex` line of `03000106` becomes four payload bytes:

```text
03 00 01 06
```

To write four weights starting at address zero:

```text
01 03 00 00 00 04 ww ww ww ww
```

The complete inference transaction is:

```text
1. WRITE target 0: imem.hex (big-endian four-byte words)
2. WRITE target 3: weight.hex
3. WRITE target 4: bias.hex
4. WRITE target 1: input.hex
5. Optionally READ selected memory locations to verify loading.
6. Send START (0x03).
7. Poll STATUS until done or error.
8. READ target 5 for model_config.json output_len bytes.
9. Compare returned bytes with golden_output.hex.
```

## Required UART hardening before board use

The existing UART RTL is a useful prototype but should be completed before it
is trusted for model deployment:

1. Reject invalid target IDs, out-of-range addresses, zero/oversized lengths,
   IMEM lengths not divisible by four, and a `START` received while busy.
2. Add a response packet/ACK after every write, including success or an error
   code.  Without an ACK, the host cannot detect a lost byte.
3. Add packet framing (for example `0xA5 0x5A`) and CRC-16, then discard a bad
   packet without changing memory.
4. Chunk long uploads (128–512 payload bytes) and wait for ACK/CRC after every
   chunk.  The HAR model currently has 1120 weight bytes and 54 bias bytes.
5. Implement four-byte IMEM readback.  The current read path returns only
   `host_rd_data32[7:0]`, which is insufficient for verifying instruction
   memory.
6. Verify the BRAM read latency in hardware; UART read sequencing assumes the
   registered one-cycle latency used by the RTL memories.
7. Do not permit parameter or program writes while `busy` is asserted.  The
   memory controller gates most such writes, but the UART interface must also
   report that rejection to the host.

## Bring-up and verification plan

1. Use the dense-only baseline and run the existing compiler, golden model,
   and top-level simulation tests.
2. Synthesize the board wrapper and inspect inferred RAM: IMEM is 32-bit;
   parameter/activation/output memories are signed 8-bit.
3. Simulate UART loading of instruction, weights, biases, and input, followed
   by start/status/output readback.
4. Write `sw/uart_loader.py` using `pyserial`.  It should parse the hex files,
   convert them to bytes, pack IMEM words big-endian, upload in ACKed chunks,
   issue START, poll status, read target 5, and compare to golden output.
5. Program the FPGA at 115200 baud and first verify a few addresses by
   readback, then run one complete inference.
6. Compare all output bytes to `golden_output.hex`; only then increase baud
   rate or optimize memory/datapath timing.

