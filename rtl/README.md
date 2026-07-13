# RTL layout

Use SystemVerilog (`.sv`) for new source files. Keep one synthesizable module per file and put shared widths/opcodes in `rtl/tinyml_pkg.sv`.

```text
rtl/
  tinyml_pkg.sv
  frontend/  program_counter, instruction_memory, instruction_fetch,
             instruction_decoder, controller_fsm
  memory/    scratchpad_memory, weight_memory, bias_memory, output_memory,
             memory_controller
  compute/   dense_engine, vector_loader, simd_mac_array, accumulator,
             bias_adder, activation_unit, requantizer, writeback_unit
  regfile/   register_file
  host_if/   uart_interface                 # last; simulation uses direct preload
  top/       tinyml_accelerator_top
```

`dense_engine` is the compute-side sub-FSM and owns the inner neuron/chunk loops. `controller_fsm` only dispatches complete decoded instructions and waits for `done`.

Detailed interfaces and build order: [docs/implementation.md](../docs/implementation.md).
