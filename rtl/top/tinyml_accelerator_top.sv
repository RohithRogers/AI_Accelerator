// =============================================================================
// tinyml_accelerator_top.sv — Top-Level TinyML Accelerator Module
//
// PURPOSE:
//   Integrates the Frontend (PC, Fetch, Decoder, Controller FSM), Memory Subsystem
//   (Instruction, Scratchpad, Weight, Bias, Output, Host Data Memories and Arbiter),
//   and Compute Engine (Dense Engine with SIMD MAC array, Accumulator, Bias Adder,
//   Requantizer, Activation, and Writeback units).
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tinyml_accelerator_top #(
  parameter int unsigned IMEM_DEPTH_WORDS = 1024,
  parameter int unsigned SPAD_DEPTH       = 4096,
  parameter int unsigned PARAM_DEPTH      = 65536,
  parameter int unsigned OUT_DEPTH        = 4096,
  parameter int unsigned HMEM_DEPTH       = 4096,
  parameter int unsigned SIMD_WIDTH       = 4,
  parameter int unsigned ACC_WIDTH        = 32,
  parameter int unsigned REQUANT_SHIFT    = 0,
  parameter int unsigned RELU6_MAX        = 127
) (
  input  logic clk,
  input  logic rst_n,

  // ---- Execution Control Interface ------------------------------------------
  input  logic        start,             // pulse to start execution from PC=0
  output logic        busy,              // high while executing
  output logic        done,              // 1-cycle pulse upon completion
  output logic        error,             // asserted on illegal state or error
  output logic [7:0]  error_code,        // error code details

  // ---- Host / Memory Preload & Readback Interface ---------------------------
  input  logic        host_wr_en,
  input  logic        host_rd_en,
  input  logic [2:0]  host_target_id,    // 0:IMEM, 1:HMEM, 2:SPAD, 3:WEIGHT, 4:BIAS, 5:OUTMEM
  input  logic [15:0] host_addr,
  input  logic [31:0] host_wr_data32,    // for 32-bit IMEM writes
  input  logic signed [7:0] host_wr_data8, // for 8-bit memory writes
  output logic [31:0] host_rd_data32,    // from 32-bit IMEM reads
  output logic signed [7:0] host_rd_data8   // from 8-bit memory reads
);

  // ---------------------------------------------------------------------------
  // Internal Interconnects
  // ---------------------------------------------------------------------------

  // Program Counter <-> Fetch / Controller
  logic        pc_write;
  logic [15:0] pc_next;
  logic [15:0] pc_out;

  // Fetch <-> Controller / Decoder
  logic        fetch_start;
  logic        instr_valid;
  logic        fetch_error;
  logic [31:0] w0, w1, w2, w3, w4;
  logic [15:0] fetch_next_pc;

  // Decoder <-> Controller
  decoded_instr_t decoded_instr_raw;

  // Instruction Memory Interconnects
  logic        imem_rd_en;
  logic [15:0] imem_rd_addr;
  logic [31:0] imem_rd_data;

  logic        imem_host_wr_en;
  logic [9:0]  imem_host_wr_addr;
  logic [31:0] imem_host_wr_data;
  logic        imem_host_rd_en;
  logic [9:0]  imem_host_rd_addr;
  logic [31:0] imem_host_rd_data;

  // Controller <-> Dense Engine Interface
  logic        dense_start;
  logic        dense_busy;
  logic        dense_done;
  logic        dense_error;
  logic [15:0] dense_input_addr;
  logic [15:0] dense_weight_addr;
  logic [15:0] dense_output_addr;
  logic [15:0] dense_bias_addr;
  logic [15:0] dense_input_len;
  logic [15:0] dense_output_len;
  logic [7:0]  dense_activation;
  logic [31:0] dense_requant_m0;
  logic signed [7:0] dense_requant_shift;

  // Convolution config + im2col interconnects
  logic        cfg_wr_en;
  logic [7:0]  cfg_img_width;
  logic [7:0]  cfg_img_height;
  logic [7:0]  cfg_channels;
  logic [7:0]  cfg_out_channels;
  logic [7:0]  cfg_kernel_h;
  logic [7:0]  cfg_kernel_w;
  logic [3:0]  cfg_stride;
  logic [3:0]  cfg_padding;
  logic [7:0]  ccfg_img_width;
  logic [7:0]  ccfg_img_height;
  logic [7:0]  ccfg_channels;
  logic [7:0]  ccfg_out_channels;
  logic [7:0]  ccfg_kernel_h;
  logic [7:0]  ccfg_kernel_w;
  logic [3:0]  ccfg_stride;
  logic [3:0]  ccfg_padding;
  logic        conv_start;
  logic        conv_busy;
  logic        conv_done;
  logic [15:0] conv_input_base;
  logic [15:0] conv_output_base;
  logic        im2col_spad_rd_en;
  logic [15:0] im2col_spad_rd_addr;
  logic signed [7:0] im2col_spad_rd_data;
  logic        im2col_spad_wr_en;
  logic [15:0] im2col_spad_wr_addr;
  logic signed [7:0] im2col_spad_wr_data;

  // Memory Controller Host Ports
  logic        hmem_host_wr_en;
  logic [11:0] hmem_host_wr_addr;
  logic signed [7:0] hmem_host_wr_data;
  logic        hmem_host_rd_en;
  logic [11:0] hmem_host_rd_addr;
  logic signed [7:0] hmem_host_rd_data;

  logic        spad_host_wr_en;
  logic [11:0] spad_host_wr_addr;
  logic signed [7:0] spad_host_wr_data;
  logic        spad_host_rd_en;
  logic [11:0] spad_host_rd_addr;
  logic signed [7:0] spad_host_rd_data;

  logic        weight_host_wr_en;
  logic [15:0] weight_host_wr_addr;
  logic signed [7:0] weight_host_wr_data;
  logic        weight_host_rd_en;
  logic [15:0] weight_host_rd_addr;
  logic signed [7:0] weight_host_rd_data;

  logic        bias_host_wr_en;
  logic [15:0] bias_host_wr_addr;
  logic signed [7:0] bias_host_wr_data;
  logic        bias_host_rd_en;
  logic [15:0] bias_host_rd_addr;
  logic signed [7:0] bias_host_rd_data;

  logic        outmem_host_rd_en;
  logic [11:0] outmem_host_rd_addr;
  logic signed [7:0] outmem_host_rd_data;

  // Datapath Memory Ports (Controller FSM & Dense Engine)
  logic        ctrl_spad_wr_en;
  logic [15:0] ctrl_spad_wr_addr;
  logic signed [7:0] ctrl_spad_wr_data;
  logic        ctrl_spad_rd_en;
  logic [15:0] ctrl_spad_rd_addr;
  logic signed [7:0] ctrl_spad_rd_data;

  logic        dense_spad_wr_en;
  logic [15:0] dense_spad_wr_addr;
  logic signed [7:0] dense_spad_wr_data;
  logic [SIMD_WIDTH*16-1:0] dense_spad_rd_addr_bus;
  logic signed [SIMD_WIDTH*8-1:0] dense_spad_rd_data_bus;

  logic [SIMD_WIDTH*16-1:0] dense_weight_rd_addr_bus;
  logic signed [SIMD_WIDTH*8-1:0] dense_weight_rd_data_bus;

  logic [15:0] dense_bias_rd_addr;
  logic signed [7:0] dense_bias_rd_data;

  logic        hmem_rd_en;
  logic [15:0] hmem_rd_addr;
  logic signed [7:0] hmem_rd_data;

  logic        omem_wr_en;
  logic [15:0] omem_wr_addr;
  logic signed [7:0] omem_wr_data;

  // ---------------------------------------------------------------------------
  // 1. Frontend Subsystem
  // ---------------------------------------------------------------------------

  program_counter u_pc (
    .clk      (clk),
    .rst_n    (rst_n),
    .pc_write (pc_write),
    .pc_next  (pc_next),
    .pc_out   (pc_out)
  );

  instruction_fetch u_fetch (
    .clk          (clk),
    .rst_n        (rst_n),
    .fetch_start  (fetch_start),
    .instr_valid  (instr_valid),
    .fetch_error  (fetch_error),
    .pc_in        (pc_out),
    .w0           (w0),
    .w1           (w1),
    .w2           (w2),
    .w3           (w3),
    .w4           (w4),
    .next_pc      (fetch_next_pc),
    .imem_rd_en   (imem_rd_en),
    .imem_rd_addr (imem_rd_addr),
    .imem_rd_data (imem_rd_data)
  );

  instruction_decoder u_decoder (
    .w0        (w0),
    .w1        (w1),
    .w2        (w2),
    .w3        (w3),
    .w4        (w4),
    .instr_out (decoded_instr_raw)
  );

  controller_fsm u_controller (
    .clk                 (clk),
    .rst_n               (rst_n),
    .start               (start),
    .busy                (busy),
    .done                (done),
    .error               (error),
    .error_code          (error_code),

    .fetch_start         (fetch_start),
    .instr_valid         (instr_valid),
    .fetch_error         (fetch_error),
    .fetch_next_pc       (fetch_next_pc),
    .w0                  (w0),
    .w1                  (w1),
    .w2                  (w2),
    .w3                  (w3),
    .instr               (decoded_instr_raw),

    .pc_write            (pc_write),
    .pc_next             (pc_next),
    .pc_out              (pc_out),

    .dense_start         (dense_start),
    .dense_busy          (dense_busy),
    .dense_done          (dense_done),
    .dense_error         (dense_error),
    .dense_input_addr    (dense_input_addr),
    .dense_weight_addr   (dense_weight_addr),
    .dense_output_addr   (dense_output_addr),
    .dense_bias_addr     (dense_bias_addr),
    .dense_input_len     (dense_input_len),
    .dense_output_len    (dense_output_len),
    .dense_activation    (dense_activation),
    .dense_requant_m0    (dense_requant_m0),
    .dense_requant_shift (dense_requant_shift),

    .spad_wr_en          (ctrl_spad_wr_en),
    .spad_wr_addr        (ctrl_spad_wr_addr),
    .spad_wr_data        (ctrl_spad_wr_data),
    .spad_rd_en          (ctrl_spad_rd_en),
    .spad_rd_addr        (ctrl_spad_rd_addr),
    .spad_rd_data        (ctrl_spad_rd_data),

    .hmem_rd_en          (hmem_rd_en),
    .hmem_rd_addr        (hmem_rd_addr),
    .hmem_rd_data        (hmem_rd_data),

    .omem_wr_en          (omem_wr_en),
    .omem_wr_addr        (omem_wr_addr),
    .omem_wr_data        (omem_wr_data),

    .cfg_wr_en           (cfg_wr_en),
    .cfg_img_width       (cfg_img_width),
    .cfg_img_height      (cfg_img_height),
    .cfg_channels        (cfg_channels),
    .cfg_out_channels    (cfg_out_channels),
    .cfg_kernel_h        (cfg_kernel_h),
    .cfg_kernel_w        (cfg_kernel_w),
    .cfg_stride          (cfg_stride),
    .cfg_padding         (cfg_padding),
    .ccfg_channels       (ccfg_channels),
    .ccfg_out_channels   (ccfg_out_channels),
    .ccfg_kernel_h       (ccfg_kernel_h),
    .ccfg_kernel_w       (ccfg_kernel_w),
    .conv_start          (conv_start),
    .conv_busy           (conv_busy),
    .conv_done           (conv_done),
    .conv_input_base     (conv_input_base),
    .conv_output_base    (conv_output_base)
  );

  // ---------------------------------------------------------------------------
  // 2. Memory Subsystem & Interconnect Arbiter
  // ---------------------------------------------------------------------------

  memory_controller u_mem_controller (
    .clk                (clk),
    .rst_n              (rst_n),
    .busy               (busy),

    .host_wr_en         (host_wr_en),
    .host_rd_en         (host_rd_en),
    .host_target_id     (host_target_id),
    .host_addr          (host_addr),
    .host_wr_data32     (host_wr_data32),
    .host_wr_data8      (host_wr_data8),
    .host_rd_data32     (host_rd_data32),
    .host_rd_data8      (host_rd_data8),

    .imem_host_wr_en    (imem_host_wr_en),
    .imem_host_wr_addr  (imem_host_wr_addr),
    .imem_host_wr_data  (imem_host_wr_data),
    .imem_host_rd_en    (imem_host_rd_en),
    .imem_host_rd_addr  (imem_host_rd_addr),
    .imem_host_rd_data  (imem_host_rd_data),

    .hmem_host_wr_en    (hmem_host_wr_en),
    .hmem_host_wr_addr  (hmem_host_wr_addr),
    .hmem_host_wr_data  (hmem_host_wr_data),
    .hmem_host_rd_en    (hmem_host_rd_en),
    .hmem_host_rd_addr  (hmem_host_rd_addr),
    .hmem_host_rd_data  (hmem_host_rd_data),

    .spad_host_wr_en    (spad_host_wr_en),
    .spad_host_wr_addr  (spad_host_wr_addr),
    .spad_host_wr_data  (spad_host_wr_data),
    .spad_host_rd_en    (spad_host_rd_en),
    .spad_host_rd_addr  (spad_host_rd_addr),
    .spad_host_rd_data  (spad_host_rd_data),

    .weight_host_wr_en  (weight_host_wr_en),
    .weight_host_wr_addr(weight_host_wr_addr),
    .weight_host_wr_data(weight_host_wr_data),
    .weight_host_rd_en  (weight_host_rd_en),
    .weight_host_rd_addr(weight_host_rd_addr),
    .weight_host_rd_data(weight_host_rd_data),

    .bias_host_wr_en    (bias_host_wr_en),
    .bias_host_wr_addr  (bias_host_wr_addr),
    .bias_host_wr_data  (bias_host_wr_data),
    .bias_host_rd_en    (bias_host_rd_en),
    .bias_host_rd_addr  (bias_host_rd_addr),
    .bias_host_rd_data  (bias_host_rd_data),

    .outmem_host_rd_en  (outmem_host_rd_en),
    .outmem_host_rd_addr(outmem_host_rd_addr),
    .outmem_host_rd_data(outmem_host_rd_data)
  );

  // --- Instruction Memory (Single Port synchronous RAM) ---
  logic                                imem_mux_rd_en;
  logic [$clog2(IMEM_DEPTH_WORDS)-1:0] imem_mux_rd_addr;

  // The execution port has priority.  Host writes are already blocked by the
  // memory controller while busy, so this avoids using the controller's busy
  // output as a combinational RAM-port mux select.
  assign imem_mux_rd_en   = imem_rd_en | imem_host_rd_en;
  assign imem_mux_rd_addr = imem_rd_en ? imem_rd_addr[$clog2(IMEM_DEPTH_WORDS)-1:0] :
                                       imem_host_rd_addr[$clog2(IMEM_DEPTH_WORDS)-1:0];

  instruction_memory #(
    .DEPTH(IMEM_DEPTH_WORDS)
  ) u_imem (
    .clk     (clk),
    .rst_n   (rst_n),
    .rd_en   (imem_mux_rd_en),
    .rd_addr (imem_mux_rd_addr),
    .rd_data (imem_rd_data),
    .wr_en   (imem_host_wr_en),
    .wr_addr (imem_host_wr_addr[$clog2(IMEM_DEPTH_WORDS)-1:0]),
    .wr_data (imem_host_wr_data)
  );
  assign imem_host_rd_data = imem_rd_data;

  // --- Scratchpad Memory (SIMD_WIDTH Replicated Banks for Parallel Read Lanes) ---
  logic        spad_mux_a_wr_en;
  logic [15:0] spad_mux_a_wr_addr;
  logic signed [7:0] spad_mux_a_wr_data;

  assign spad_mux_a_wr_en   = dense_spad_wr_en ? dense_spad_wr_en :
                             im2col_spad_wr_en ? im2col_spad_wr_en : ctrl_spad_wr_en;
  assign spad_mux_a_wr_addr = dense_spad_wr_en ? dense_spad_wr_addr :
                             im2col_spad_wr_en ? im2col_spad_wr_addr : ctrl_spad_wr_addr;
  assign spad_mux_a_wr_data = dense_spad_wr_en ? dense_spad_wr_data :
                             im2col_spad_wr_en ? im2col_spad_wr_data : ctrl_spad_wr_data;

  logic signed [7:0] spad_lane_rd_data [0:SIMD_WIDTH-1];

  generate
    for (genvar lane = 0; lane < SIMD_WIDTH; lane++) begin : g_spad_banks
      logic [15:0] lane_rd_addr;
      if (lane == 0) begin : g_lane0_addr
        assign lane_rd_addr = dense_busy ? dense_spad_rd_addr_bus[15:0] :
                              conv_busy  ? im2col_spad_rd_addr : ctrl_spad_rd_addr;
      end else begin : g_lanen_addr
        assign lane_rd_addr = dense_spad_rd_addr_bus[lane*16 +: 16];
      end

      scratchpad_memory #(
        .DEPTH(SPAD_DEPTH)
      ) u_spad_bank (
        .clk       (clk),
        .rst_n     (rst_n),

        // Port A: Datapath Compute / Controller FSM
        .a_rd_en   (1'b1),
        .a_rd_addr (lane_rd_addr[$clog2(SPAD_DEPTH)-1:0]),
        .a_rd_data (spad_lane_rd_data[lane]),
        .a_wr_en   (spad_mux_a_wr_en),
        .a_wr_addr (spad_mux_a_wr_addr[$clog2(SPAD_DEPTH)-1:0]),
        .a_wr_data (spad_mux_a_wr_data),

        // Port B: Host / UART Interface
        .b_rd_en   (spad_host_rd_en),
        .b_rd_addr (spad_host_rd_addr[$clog2(SPAD_DEPTH)-1:0]),
        .b_rd_data (),
        .b_wr_en   (spad_host_wr_en),
        .b_wr_addr (spad_host_wr_addr[$clog2(SPAD_DEPTH)-1:0]),
        .b_wr_data (spad_host_wr_data)
      );

      assign dense_spad_rd_data_bus[lane*8 +: 8] = spad_lane_rd_data[lane];
    end
  endgenerate

  assign ctrl_spad_rd_data   = spad_lane_rd_data[0];
  assign im2col_spad_rd_data = spad_lane_rd_data[0];
  assign spad_host_rd_data   = spad_lane_rd_data[0];

  // --- Weight Memory (SIMD_WIDTH Replicated Banks for Parallel Read Lanes) ---
  logic signed [7:0] weight_lane_rd_data [0:SIMD_WIDTH-1];

  generate
    for (genvar lane = 0; lane < SIMD_WIDTH; lane++) begin : g_weight_banks
      logic [15:0] wt_lane_rd_addr;
      if (lane == 0) begin : g_wt_lane0
        assign wt_lane_rd_addr = dense_busy ? dense_weight_rd_addr_bus[15:0] : weight_host_rd_addr;
      end else begin : g_wt_lanen
        assign wt_lane_rd_addr = dense_weight_rd_addr_bus[lane*16 +: 16];
      end

      weight_memory #(
        .DEPTH(PARAM_DEPTH)
      ) u_weight_bank (
        .clk     (clk),
        .rst_n   (rst_n),
        .rd_en   (1'b1),
        .rd_addr (wt_lane_rd_addr[$clog2(PARAM_DEPTH)-1:0]),
        .rd_data (weight_lane_rd_data[lane]),
        .wr_en   (weight_host_wr_en),
        .wr_addr (weight_host_wr_addr[$clog2(PARAM_DEPTH)-1:0]),
        .wr_data (weight_host_wr_data)
      );

      assign dense_weight_rd_data_bus[lane*8 +: 8] = weight_lane_rd_data[lane];
    end
  endgenerate

  assign weight_host_rd_data = weight_lane_rd_data[0];

  // --- Bias Memory ---
  logic [15:0] bias_mux_rd_addr;
  assign bias_mux_rd_addr = dense_busy ? dense_bias_rd_addr : bias_host_rd_addr;

  bias_memory #(
    .DEPTH(PARAM_DEPTH)
  ) u_bias_mem (
    .clk     (clk),
    .rst_n   (rst_n),
    .rd_en   (1'b1),
    .rd_addr (bias_mux_rd_addr[$clog2(PARAM_DEPTH)-1:0]),
    .rd_data (dense_bias_rd_data),
    .wr_en   (bias_host_wr_en),
    .wr_addr (bias_host_wr_addr[$clog2(PARAM_DEPTH)-1:0]),
    .wr_data (bias_host_wr_data)
  );

  assign bias_host_rd_data = dense_bias_rd_data;

  // --- Host Data Memory (Source for LOAD, destination for STORE/Readback) ---
  host_data_memory #(
    .DEPTH(HMEM_DEPTH)
  ) u_hmem (
    .clk       (clk),
    .rst_n     (rst_n),

    // Port A: Accelerator Datapath (LOAD source)
    .a_rd_en   (hmem_rd_en),
    .a_rd_addr (hmem_rd_addr[$clog2(HMEM_DEPTH)-1:0]),
    .a_rd_data (hmem_rd_data),
    .a_wr_en   (1'b0),
    .a_wr_addr ({$clog2(HMEM_DEPTH){1'b0}}),
    .a_wr_data (8'sd0),

    // Port B: Host / UART Interface
    .b_rd_en   (hmem_host_rd_en),
    .b_rd_addr (hmem_host_rd_addr[$clog2(HMEM_DEPTH)-1:0]),
    .b_rd_data (hmem_host_rd_data),
    .b_wr_en   (hmem_host_wr_en),
    .b_wr_addr (hmem_host_wr_addr[$clog2(HMEM_DEPTH)-1:0]),
    .b_wr_data (hmem_host_wr_data)
  );

  // --- Output Memory (Written by Accelerator STORE, read by Host) ---
  output_memory #(
    .DEPTH(OUT_DEPTH)
  ) u_outmem (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (omem_wr_en),
    .wr_addr (omem_wr_addr[$clog2(OUT_DEPTH)-1:0]),
    .wr_data (omem_wr_data),
    .rd_en   (outmem_host_rd_en),
    .rd_addr (outmem_host_rd_addr[$clog2(OUT_DEPTH)-1:0]),
    .rd_data (outmem_host_rd_data)
  );

  // ---------------------------------------------------------------------------
  // 3. Compute Subsystem
  // ---------------------------------------------------------------------------

  conv_config_reg u_conv_cfg (
    .clk            (clk),
    .rst_n          (rst_n),
    .cfg_wr_en      (cfg_wr_en),
    .cfg_img_width  (cfg_img_width),
    .cfg_img_height (cfg_img_height),
    .cfg_channels   (cfg_channels),
    .cfg_out_channels(cfg_out_channels),
    .cfg_kernel_h   (cfg_kernel_h),
    .cfg_kernel_w   (cfg_kernel_w),
    .cfg_stride     (cfg_stride),
    .cfg_padding    (cfg_padding),
    .out_img_width  (ccfg_img_width),
    .out_img_height (ccfg_img_height),
    .out_channels   (ccfg_channels),
    .out_out_channels(ccfg_out_channels),
    .out_kernel_h    (ccfg_kernel_h),
    .out_kernel_w    (ccfg_kernel_w),
    .out_stride     (ccfg_stride),
    .out_padding    (ccfg_padding)
  );

  im2col_unit #(
    .SPAD_DEPTH(SPAD_DEPTH)
  ) u_im2col (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (conv_start),
    .busy         (conv_busy),
    .done         (conv_done),
    .img_width    (ccfg_img_width),
    .img_height   (ccfg_img_height),
    .channels     (ccfg_channels),
    .kernel_h     (ccfg_kernel_h),
    .kernel_w     (ccfg_kernel_w),
    .stride       ({4'h0, ccfg_stride}),
    .padding      ({4'h0, ccfg_padding}),
    .input_base   (conv_input_base),
    .output_base  (conv_output_base),
    .spad_rd_en   (im2col_spad_rd_en),
    .spad_rd_addr (im2col_spad_rd_addr),
    .spad_rd_data (im2col_spad_rd_data),
    .spad_wr_en   (im2col_spad_wr_en),
    .spad_wr_addr (im2col_spad_wr_addr),
    .spad_wr_data (im2col_spad_wr_data)
  );

  dense_engine #(
    .SIMD_WIDTH    (SIMD_WIDTH),
    .ACC_WIDTH     (ACC_WIDTH),
    .REQUANT_SHIFT (REQUANT_SHIFT),
    .RELU6_MAX     (RELU6_MAX)
  ) u_dense_engine (
    .clk                (clk),
    .rst_n              (rst_n),

    .start              (dense_start),
    .input_addr         (dense_input_addr),
    .weight_addr        (dense_weight_addr),
    .output_addr        (dense_output_addr),
    .bias_addr          (dense_bias_addr),
    .input_len          (dense_input_len),
    .output_len         (dense_output_len),
    .activation         (dense_activation),
    .requant_m0         (dense_requant_m0),
    .requant_shift      (dense_requant_shift),

    .busy               (dense_busy),
    .done               (dense_done),
    .error              (dense_error),

    .spad_rd_addr_bus   (dense_spad_rd_addr_bus),
    .spad_rd_data_bus   (dense_spad_rd_data_bus),

    .spad_wr_en         (dense_spad_wr_en),
    .spad_wr_addr       (dense_spad_wr_addr),
    .spad_wr_data       (dense_spad_wr_data),

    .weight_rd_addr_bus (dense_weight_rd_addr_bus),
    .weight_rd_data_bus (dense_weight_rd_data_bus),

    .bias_rd_addr       (dense_bias_rd_addr),
    .bias_rd_data       (dense_bias_rd_data)
  );

endmodule
