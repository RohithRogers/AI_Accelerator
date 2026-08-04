// =============================================================================
// controller_fsm.sv — Top-Level Sequencing FSM
//
// PURPOSE:
//   Owns program sequencing: it is the only block that may advance the PC,
//   dispatch instructions to executors, and pulse the top-level done/error
//   signals.  The dense_engine and memory copy loops are *sub-executors*;
//   the controller only starts them and waits for their `done`.
//
// STATE SEQUENCE (registered one-hot or binary — your choice):
//
//   IDLE       : waits for top-level `start` pulse.
//   FETCH      : asserts fetch_start, waits for instr_valid from instruction_fetch.
//   DECODE     : latches decoded_instr_t from instruction_decoder (one cycle).
//   DISPATCH   : checks opcode and bounds; routes to the correct executor state.
//                Goes to ERROR on decode_error, bad opcode, or failed bounds check.
//   LOAD_COPY  : iterates a 16-bit `remaining` counter, copying bytes one-per-cycle
//                from host-data memory to scratchpad.  Increments src/dst addresses.
//   STORE_COPY : same as LOAD_COPY but scratchpad -> output memory direction.
//   DENSE_START: asserts dense_engine `start` for one cycle with all operands stable.
//   DENSE_WAIT : waits for dense_engine `done` or `error`.
//   ACT_LOOP   : iterates activation in-place on scratchpad (or delegates to activation_unit).
//   NOP        : one idle cycle, then RETIRE.
//   RETIRE     : asserts `pc_write` with the latched `next_pc`; then goes back to FETCH.
//   DONE       : pulses top-level `done` for one cycle; then goes to IDLE.
//   ERROR      : asserts top-level `error` + `error_code`; holds until rst_n or policy allows restart.
//
// KEY RULES:
//   - PC advances ONLY in RETIRE.
//   - `done` is exactly one clock wide; `busy` is high from accepted start through RETIRE.
//   - END goes DISPATCH -> END_STATE -> DONE -> IDLE (no RETIRE, PC does not advance).
//   - Any executor error routes to ERROR with a distinctive error_code.
//
// CONNECTIONS (see implementation.md for full width table):
//   Top-level  : start, busy, done, error, error_code
//   Fetch unit : fetch_start, instr_valid, fetch_error, w0..w3, next_pc_from_fetch
//   Decoder    : instr_out (decoded_instr_t) — purely combinational, always valid
//   PC         : pc_write, pc_next
//   Dense eng  : dense_start, dense_busy, dense_done, dense_error
//                + input_addr, weight_addr, output_addr, bias_addr,
//                  input_len, output_len, activation (held stable until done)
//   Memories   : scratchpad rd/wr port, output_mem wr port, host_mem rd port
//                (for LOAD_COPY / STORE_COPY byte loops)
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module controller_fsm (
  input  logic clk,
  input  logic rst_n,

  // ---- Top-level execution interface ----------------------------------------
  input  logic        start,             // sampled only while in IDLE
  output logic        busy,              // high from FETCH through RETIRE
  output logic        done,              // one-cycle pulse in DONE state
  output logic        error,             // asserted in ERROR state
  output logic [7:0]  error_code,        // distinguishes error source

  // ---- Instruction fetch handshake ------------------------------------------
  output logic        fetch_start,       // pulse to begin a fetch
  input  logic        instr_valid,       // one-cycle pulse: words assembled
  input  logic        fetch_error,       // illegal header opcode from fetch unit
  input  logic [31:0] w0, w1, w2, w3,   // raw word bundle for decoder

  // ---- Decoded instruction (combinational from instruction_decoder) ----------
  input  decoded_instr_t instr,          // always-valid combinational decode of w0..w3

  // ---- Program counter control ----------------------------------------------
  output logic        pc_write,          // strobe: commit next_pc to PC register
  output logic [15:0] pc_next,           // next_pc value (latched from fetch)
  input  logic [15:0] pc_out,            // current PC (from program_counter)

  // ---- Dense engine command interface ---------------------------------------
  output logic        dense_start,       // one-cycle start pulse
  input  logic        dense_busy,
  input  logic        dense_done,        // one-cycle done pulse
  input  logic        dense_error,
  output logic [15:0] dense_input_addr,
  output logic [15:0] dense_weight_addr,
  output logic [15:0] dense_output_addr,
  output logic [15:0] dense_bias_addr,
  output logic [15:0] dense_input_len,
  output logic [15:0] dense_output_len,
  output logic [7:0]  dense_activation,

  // ---- Scratchpad read/write (LOAD_COPY dst / ACT_LOOP / STORE_COPY src) ----
  output logic        spad_wr_en,
  output logic [15:0] spad_wr_addr,
  output logic signed [7:0] spad_wr_data,
  output logic        spad_rd_en,
  output logic [15:0] spad_rd_addr,
  input  logic signed [7:0] spad_rd_data,

  // ---- Host-data memory read (LOAD_COPY source) -----------------------------
  output logic        hmem_rd_en,
  output logic [15:0] hmem_rd_addr,
  input  logic signed [7:0] hmem_rd_data,

  // ---- Output memory write (STORE_COPY destination) -------------------------
  output logic        omem_wr_en,
  output logic [15:0] omem_wr_addr,
  output logic signed [7:0] omem_wr_data
);

  // TODO: declare FSM state enum (IDLE, FETCH, DECODE, DISPATCH, LOAD_COPY,
  //       STORE_COPY, DENSE_START, DENSE_WAIT, ACT_LOOP, NOP_ST, RETIRE,
  //       DONE_ST, END_ST, ERROR_ST)

  // TODO: declare state register (current_state, next_state)

  // TODO: declare working registers:
  //   - latched decoded_instr_t (latched_instr) captured in DECODE
  //   - next_pc_latch (16-bit, captured when instr_valid)
  //   - copy_src_addr, copy_dst_addr, copy_remaining (16-bit, for LOAD/STORE loops)

  // TODO: implement sequential block (always_ff) for state register and working regs

  // TODO: implement combinational block (always_comb) for:
  //   - next state logic (all state transitions per header comment)
  //   - output driving (fetch_start, pc_write, dense_start, spad/hmem/omem ports, etc.)

endmodule
