// =============================================================================
// host_data_memory.sv — Host Data Memory Model / Stream Buffer
//
// PURPOSE:
//   Signed INT8 byte memory representing host RAM. Source for `LOAD` instructions
//   and destination for `STORE` instructions when communicating with host model/UART.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module host_data_memory #(
  parameter int unsigned DEPTH = 4096
) (
  input  logic clk,
  input  logic rst_n,

  // Port A (Accelerator controller LOAD source / STORE destination)
  input  logic a_rd_en,
  input  logic [$clog2(DEPTH)-1:0] a_rd_addr,
  output logic signed [7:0] a_rd_data,
  input  logic a_wr_en,
  input  logic [$clog2(DEPTH)-1:0] a_wr_addr,
  input  logic signed [7:0] a_wr_data,

  // Port B (Host / UART write/read access)
  input  logic b_rd_en,
  input  logic [$clog2(DEPTH)-1:0] b_rd_addr,
  output logic signed [7:0] b_rd_data,
  input  logic b_wr_en,
  input  logic [$clog2(DEPTH)-1:0] b_wr_addr,
  input  logic signed [7:0] b_wr_data
);

  logic signed [7:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (a_wr_en) begin
      mem[a_wr_addr] <= a_wr_data;
    end
    if (a_rd_en) begin
      a_rd_data <= mem[a_rd_addr];
    end
  end

  always_ff @(posedge clk) begin
    if (b_wr_en) begin
      mem[b_wr_addr] <= b_wr_data;
    end
    if (b_rd_en) begin
      b_rd_data <= mem[b_rd_addr];
    end
  end

endmodule
