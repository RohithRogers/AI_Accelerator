// =============================================================================
// output_memory.sv — Output Result Memory
//
// PURPOSE:
//   Signed INT8 byte memory holding final output tensors written by `STORE`.
//   Host/UART bridge reads results after `done` is pulsed.
//   1-cycle registered read latency.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module output_memory #(
  parameter int unsigned DEPTH = OUT_DEPTH
) (
  input  logic clk,
  input  logic rst_n,

  // Write port (Accelerator controller STORE instruction)
  input  logic wr_en,
  input  logic [$clog2(DEPTH)-1:0] wr_addr,
  input  logic signed [7:0] wr_data,

  // Read port (Host / UART readback)
  input  logic rd_en,
  input  logic [$clog2(DEPTH)-1:0] rd_addr,
  output logic signed [7:0] rd_data
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
