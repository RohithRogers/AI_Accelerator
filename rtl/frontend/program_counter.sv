// =============================================================================
// program_counter.sv — Program Counter
//
// PURPOSE:
//   Holds and advances the current instruction word address (in 32-bit words).
//   The PC is committed ("retired") only when the controller_fsm explicitly
//   asserts `pc_write`.  This prevents speculative advances — the fetch unit
//   reads the address, computes next_pc, and hands that back; the controller
//   decides when to actually move the pointer.
//
// BEHAVIOR:
//   - Synchronous active-low reset: on rst_n=0, PC resets to 0.
//   - When `pc_write` is asserted, `pc_next` is loaded into the register.
//   - `pc_out` is continuously driven from the register (registered, not comb).
//   - Addresses are 16-bit, covering up to IMEM_DEPTH_WORDS words.
//
// CONNECTIONS:
//   Driven by  : controller_fsm (provides pc_next and pc_write strobe)
//   Drives     : instruction_fetch (consumes pc_out as the read address)
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module program_counter (
  input  logic        clk,
  input  logic        rst_n,

  // Write port — from controller_fsm RETIRE state
  input  logic        pc_write,           // strobe: load pc_next on this cycle
  input  logic [15:0] pc_next,            // new PC value (next_pc from fetch)

  // Read port — to instruction_fetch
  output logic [15:0] pc_out              // current PC (registered)
);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pc_out <= 16'h0000;
    end else if (pc_write) begin
      pc_out <= pc_next;
    end
  end

endmodule
