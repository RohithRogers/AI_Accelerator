// =============================================================================
// instruction_memory.sv — Instruction Memory (32-bit word-addressed ROM/RAM)
//
// PURPOSE:
//   Single-port synchronous memory that stores the program as 32-bit words.
//   During simulation the testbench (or host interface) preloads it through
//   the write port before asserting `start`.  During execution instruction_fetch
//   drives the read port and the memory is effectively read-only.
//
// TIMING:
//   One-cycle read latency: `rd_data` is registered and valid one cycle after
//   `rd_en` is asserted with a valid `rd_addr`.  instruction_fetch must account
//   for this latency when sequencing multi-word fetches.
//
// PARAMETERS:
//   DEPTH  — number of 32-bit words (default IMEM_DEPTH_WORDS from package)
//
// CONNECTIONS:
//   Read  : instruction_fetch drives rd_en / rd_addr; reads rd_data.
//   Write : host preload path (testbench task or future UART bridge)
//           drives wr_en / wr_addr / wr_data before start.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module instruction_memory #(
  parameter int unsigned DEPTH = IMEM_DEPTH_WORDS
) (
  input  logic        clk,
  input  logic        rst_n,

  // Read port (from instruction_fetch)
  input  logic        rd_en,
  input  logic [$clog2(DEPTH)-1:0] rd_addr,
  output logic [31:0] rd_data,          // registered; valid one cycle after rd_en

  // Write port (host preload / testbench)
  input  logic        wr_en,
  input  logic [$clog2(DEPTH)-1:0] wr_addr,
  input  logic [31:0] wr_data
);

  // Storage array
  logic [31:0] mem [0:DEPTH-1];

  // Synchronous read (1-cycle latency) and write
  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_data;
    end
    if (rd_en) begin
      rd_data <= mem[rd_addr];
    end
  end

endmodule
