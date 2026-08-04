// =============================================================================
// memory_controller.sv — Memory Subsystem Interconnect & Arbiter
//
// PURPOSE:
//   Routes host/UART read and write commands to the appropriate physical memory
//   array based on `target_id` (0=IMEM, 1=HMEM, 2=SPAD, 3=WEIGHT, 4=BIAS, 5=OUTMEM).
//   When `busy` is asserted, host write access to execution memories is gated
//   to prevent corruption.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module memory_controller (
  input  logic clk,
  input  logic rst_n,

  // Status from top accelerator
  input  logic busy,

  // ---- Host / UART Interface Bus --------------------------------------------
  input  logic        host_wr_en,
  input  logic        host_rd_en,
  input  logic [2:0]  host_target_id,  // 0:IMEM, 1:HMEM, 2:SPAD, 3:WEIGHT, 4:BIAS, 5:OUTMEM
  input  logic [15:0] host_addr,
  input  logic [31:0] host_wr_data32,  // For 32-bit IMEM writes
  input  logic signed [7:0] host_wr_data8, // For 8-bit byte memory writes
  output logic [31:0] host_rd_data32,  // From IMEM
  output logic signed [7:0] host_rd_data8,  // From 8-bit byte memories

  // ---- IMEM Port (Instruction Memory) --------------------------------------
  output logic        imem_host_wr_en,
  output logic [9:0]  imem_host_wr_addr,
  output logic [31:0] imem_host_wr_data,
  output logic        imem_host_rd_en,
  output logic [9:0]  imem_host_rd_addr,
  input  logic [31:0] imem_host_rd_data,

  // ---- HMEM Port (Host Data Memory) -----------------------------------------
  output logic        hmem_host_wr_en,
  output logic [11:0] hmem_host_wr_addr,
  output logic signed [7:0] hmem_host_wr_data,
  output logic        hmem_host_rd_en,
  output logic [11:0] hmem_host_rd_addr,
  input  logic signed [7:0] hmem_host_rd_data,

  // ---- SPAD Port (Scratchpad Memory Port B) ---------------------------------
  output logic        spad_host_wr_en,
  output logic [11:0] spad_host_wr_addr,
  output logic signed [7:0] spad_host_wr_data,
  output logic        spad_host_rd_en,
  output logic [11:0] spad_host_rd_addr,
  input  logic signed [7:0] spad_host_rd_data,

  // ---- WEIGHT Port (Weight Memory) -----------------------------------------
  output logic        weight_host_wr_en,
  output logic [15:0] weight_host_wr_addr,
  output logic signed [7:0] weight_host_wr_data,
  output logic        weight_host_rd_en,
  output logic [15:0] weight_host_rd_addr,
  input  logic signed [7:0] weight_host_rd_data,

  // ---- BIAS Port (Bias Memory) ---------------------------------------------
  output logic        bias_host_wr_en,
  output logic [15:0] bias_host_wr_addr,
  output logic signed [7:0] bias_host_wr_data,
  output logic        bias_host_rd_en,
  output logic [15:0] bias_host_rd_addr,
  input  logic signed [7:0] bias_host_rd_data,

  // ---- OUTMEM Port (Output Memory) -----------------------------------------
  output logic        outmem_host_rd_en,
  output logic [11:0] outmem_host_rd_addr,
  input  logic signed [7:0] outmem_host_rd_data
);

  // Host write gating: host writes only allowed when NOT busy (except HMEM)
  logic allow_host_wr;
  assign allow_host_wr = host_wr_en && (~busy);

  // IMEM (target 0)
  assign imem_host_wr_en   = allow_host_wr && (host_target_id == 3'd0);
  assign imem_host_wr_addr = host_addr[9:0];
  assign imem_host_wr_data = host_wr_data32;
  assign imem_host_rd_en   = host_rd_en && (host_target_id == 3'd0);
  assign imem_host_rd_addr = host_addr[9:0];

  // HMEM (target 1)
  assign hmem_host_wr_en   = host_wr_en && (host_target_id == 3'd1);
  assign hmem_host_wr_addr = host_addr[11:0];
  assign hmem_host_wr_data = host_wr_data8;
  assign hmem_host_rd_en   = host_rd_en && (host_target_id == 3'd1);
  assign hmem_host_rd_addr = host_addr[11:0];

  // SPAD (target 2)
  assign spad_host_wr_en   = allow_host_wr && (host_target_id == 3'd2);
  assign spad_host_wr_addr = host_addr[11:0];
  assign spad_host_wr_data = host_wr_data8;
  assign spad_host_rd_en   = host_rd_en && (host_target_id == 3'd2);
  assign spad_host_rd_addr = host_addr[11:0];

  // WEIGHT (target 3)
  assign weight_host_wr_en   = allow_host_wr && (host_target_id == 3'd3);
  assign weight_host_wr_addr = host_addr[15:0];
  assign weight_host_wr_data = host_wr_data8;
  assign weight_host_rd_en   = host_rd_en && (host_target_id == 3'd3);
  assign weight_host_rd_addr = host_addr[15:0];

  // BIAS (target 4)
  assign bias_host_wr_en   = allow_host_wr && (host_target_id == 3'd4);
  assign bias_host_wr_addr = host_addr[15:0];
  assign bias_host_wr_data = host_wr_data8;
  assign bias_host_rd_en   = host_rd_en && (host_target_id == 3'd4);
  assign bias_host_rd_addr = host_addr[15:0];

  // OUTMEM (target 5)
  assign outmem_host_rd_en   = host_rd_en && (host_target_id == 3'd5);
  assign outmem_host_rd_addr = host_addr[11:0];

  // Read data multiplexer
  assign host_rd_data32 = imem_host_rd_data;

  always_comb begin
    case (host_target_id)
      3'd1:    host_rd_data8 = hmem_host_rd_data;
      3'd2:    host_rd_data8 = spad_host_rd_data;
      3'd3:    host_rd_data8 = weight_host_rd_data;
      3'd4:    host_rd_data8 = bias_host_rd_data;
      3'd5:    host_rd_data8 = outmem_host_rd_data;
      default: host_rd_data8 = 8'sd0;
    endcase
  end

endmodule
