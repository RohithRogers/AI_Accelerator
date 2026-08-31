// =============================================================================
// scratchpad_memory.sv — Scratchpad Memory (Activation Storage)
//
// PURPOSE:
//   Signed INT8 byte memory holding intermediate activations and layer inputs/outputs.
//   Supports dual ports (Port A for internal datapath/FSM, Port B for Host/UART bridge).
//   Reads have 1-cycle registered latency.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module scratchpad_memory #(
  parameter int unsigned DEPTH = SPAD_DEPTH
) (
  input  logic clk,
  input  logic rst_n,

  // Port A (Datapath / Controller FSM)
  input  logic a_rd_en,
  input  logic [$clog2(DEPTH)-1:0] a_rd_addr,
  output logic signed [7:0] a_rd_data,
  input  logic a_wr_en,
  input  logic [$clog2(DEPTH)-1:0] a_wr_addr,
  input  logic signed [7:0] a_wr_data,

  // Port B (Host Interface / UART bridge)
  input  logic b_rd_en,
  input  logic [$clog2(DEPTH)-1:0] b_rd_addr,
  output logic signed [7:0] b_rd_data,
  input  logic b_wr_en,
  input  logic [$clog2(DEPTH)-1:0] b_wr_addr,
  input  logic signed [7:0] b_wr_data
);

  (* ram_style = "block" *) logic signed [7:0] mem [0:DEPTH-1];

  // Synchronous True Dual-Port Read/Write (Single-clock Vivado BRAM Template)
  always_ff @(posedge clk) begin
    // Port A
    if (a_wr_en) begin
      mem[a_wr_addr] <= a_wr_data;
    end
    if (a_rd_en) begin
      a_rd_data <= mem[a_rd_addr];
    end

    // Port B
    if (b_wr_en) begin
      mem[b_wr_addr] <= b_wr_data;
    end
    if (b_rd_en) begin
      b_rd_data <= mem[b_rd_addr];
    end
  end

endmodule
