// =============================================================================
// conv_config_reg.sv — Convolution Configuration Register File
//
// PURPOSE:
//   Holds the convolution geometry written by a CONV_CFG instruction.
//   The controller FSM writes all fields in a single cycle when CONV_CFG
//   executes.  The im2col_unit reads these fields when a CONV instruction
//   subsequently fires.
//
// Fields stored (all written simultaneously from the decoded instruction):
//   img_width    — input feature map width  (pixels)
//   img_height   — input feature map height (pixels)
//   channels     — number of input channels
//   kernel_size  — square kernel side length (e.g. 3 for a 3×3 kernel)
//   stride       — sliding window step size
//   padding      — zero-padding added to each spatial edge
//
// Usage:
//   1. Controller FSM asserts cfg_wr_en for one clock cycle after decoding
//      a CONV_CFG instruction, driving all cfg_* inputs.
//   2. On the next CONV instruction, the FSM reads back the stored fields
//      and forwards them to im2col_unit.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module conv_config_reg (
    input  logic        clk,
    input  logic        rst_n,

    // Write interface (from controller FSM on CONV_CFG execution)
    input  logic        cfg_wr_en,
    input  logic [7:0]  cfg_img_width,
    input  logic [7:0]  cfg_img_height,
    input  logic [7:0]  cfg_channels,
    input  logic [3:0]  cfg_kernel_size,
    input  logic [3:0]  cfg_stride,
    input  logic [3:0]  cfg_padding,

    // Read interface (to im2col_unit on CONV execution)
    output logic [7:0]  out_img_width,
    output logic [7:0]  out_img_height,
    output logic [7:0]  out_channels,
    output logic [3:0]  out_kernel_size,
    output logic [3:0]  out_stride,
    output logic [3:0]  out_padding
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_img_width   <= 8'h00;
            out_img_height  <= 8'h00;
            out_channels    <= 8'h01;
            out_kernel_size <= 4'h1;
            out_stride      <= 4'h1;
            out_padding     <= 4'h0;
        end else if (cfg_wr_en) begin
            out_img_width   <= cfg_img_width;
            out_img_height  <= cfg_img_height;
            out_channels    <= cfg_channels;
            out_kernel_size <= cfg_kernel_size;
            out_stride      <= cfg_stride;
            out_padding     <= cfg_padding;
        end
    end

endmodule
