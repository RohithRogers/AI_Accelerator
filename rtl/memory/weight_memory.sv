// =============================================================================
// weight_memory.sv — Weight Parameter Memory
//
// PURPOSE:
//   Signed INT8 byte memory holding neural network weights.
//   Host/UART bridge writes parameters before execution; dense engine reads
//   parameters during execution with 1-cycle registered read latency.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module weight_memory #(
  parameter int unsigned DEPTH = PARAM_DEPTH
) (
  input  logic clk,
  input  logic rst_n,

  // Read port (Dense engine)
  input  logic rd_en,
  input  logic [$clog2(DEPTH)-1:0] rd_addr,
  output logic signed [7:0] rd_data,

  // Write port (Host preload / UART)
  input  logic wr_en,
  input  logic [$clog2(DEPTH)-1:0] wr_addr,
  input  logic signed [7:0] wr_data
);

  logic signed [7:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_data;
    end
    if (rd_en) begin
      rd_data <= mem[rd_addr];
    end
  end

endmodule
