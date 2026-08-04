// =============================================================================
// tb_memory.sv — Testbench for Memory Subsystem and Memory Controller
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_memory;

  logic clk;
  logic rst_n;
  logic busy;

  // Host bus
  logic        host_wr_en;
  logic        host_rd_en;
  logic [2:0]  host_target_id;
  logic [15:0] host_addr;
  logic [31:0] host_wr_data32;
  logic signed [7:0] host_wr_data8;
  logic [31:0] host_rd_data32;
  logic signed [7:0] host_rd_data8;

  // IMEM signals
  logic        imem_wr_en, imem_rd_en;
  logic [9:0]  imem_wr_addr, imem_rd_addr;
  logic [31:0] imem_wr_data, imem_rd_data;

  // HMEM signals
  logic        hmem_a_rd_en, hmem_a_wr_en, hmem_b_rd_en, hmem_b_wr_en;
  logic [11:0] hmem_a_rd_addr, hmem_a_wr_addr, hmem_b_rd_addr, hmem_b_wr_addr;
  logic signed [7:0] hmem_a_rd_data, hmem_a_wr_data, hmem_b_rd_data, hmem_b_wr_data;

  // SPAD signals
  logic        spad_a_rd_en, spad_a_wr_en, spad_b_rd_en, spad_b_wr_en;
  logic [11:0] spad_a_rd_addr, spad_a_wr_addr, spad_b_rd_addr, spad_b_wr_addr;
  logic signed [7:0] spad_a_rd_data, spad_a_wr_data, spad_b_rd_data, spad_b_wr_data;

  // WEIGHT signals
  logic        weight_rd_en, weight_wr_en;
  logic [15:0] weight_rd_addr, weight_wr_addr;
  logic signed [7:0] weight_rd_data, weight_wr_data;

  // BIAS signals
  logic        bias_rd_en, bias_wr_en;
  logic [15:0] bias_rd_addr, bias_wr_addr;
  logic signed [7:0] bias_rd_data, bias_wr_data;

  // OUTMEM signals
  logic        outmem_rd_en, outmem_wr_en;
  logic [11:0] outmem_rd_addr, outmem_wr_addr;
  logic signed [7:0] outmem_rd_data, outmem_wr_data;

  // Memory controller instantiation
  memory_controller u_mem_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .busy(busy),
    .host_wr_en(host_wr_en),
    .host_rd_en(host_rd_en),
    .host_target_id(host_target_id),
    .host_addr(host_addr),
    .host_wr_data32(host_wr_data32),
    .host_wr_data8(host_wr_data8),
    .host_rd_data32(host_rd_data32),
    .host_rd_data8(host_rd_data8),

    .imem_host_wr_en(imem_wr_en),
    .imem_host_wr_addr(imem_wr_addr),
    .imem_host_wr_data(imem_wr_data),
    .imem_host_rd_en(imem_rd_en),
    .imem_host_rd_addr(imem_rd_addr),
    .imem_host_rd_data(imem_rd_data),

    .hmem_host_wr_en(hmem_b_wr_en),
    .hmem_host_wr_addr(hmem_b_wr_addr),
    .hmem_host_wr_data(hmem_b_wr_data),
    .hmem_host_rd_en(hmem_b_rd_en),
    .hmem_host_rd_addr(hmem_b_rd_addr),
    .hmem_host_rd_data(hmem_b_rd_data),

    .spad_host_wr_en(spad_b_wr_en),
    .spad_host_wr_addr(spad_b_wr_addr),
    .spad_host_wr_data(spad_b_wr_data),
    .spad_host_rd_en(spad_b_rd_en),
    .spad_host_rd_addr(spad_b_rd_addr),
    .spad_host_rd_data(spad_b_rd_data),

    .weight_host_wr_en(weight_wr_en),
    .weight_host_wr_addr(weight_wr_addr),
    .weight_host_wr_data(weight_wr_data),
    .weight_host_rd_en(weight_rd_en),
    .weight_host_rd_addr(weight_rd_addr),
    .weight_host_rd_data(weight_rd_data),

    .bias_host_wr_en(bias_wr_en),
    .bias_host_wr_addr(bias_wr_addr),
    .bias_host_wr_data(bias_wr_data),
    .bias_host_rd_en(bias_rd_en),
    .bias_host_rd_addr(bias_rd_addr),
    .bias_host_rd_data(bias_rd_data),

    .outmem_host_rd_en(outmem_rd_en),
    .outmem_host_rd_addr(outmem_rd_addr),
    .outmem_host_rd_data(outmem_rd_data)
  );

  // Physical memories
  instruction_memory #(.DEPTH(1024)) u_imem (
    .clk(clk), .rst_n(rst_n),
    .rd_en(imem_rd_en), .rd_addr(imem_rd_addr), .rd_data(imem_rd_data),
    .wr_en(imem_wr_en), .wr_addr(imem_wr_addr), .wr_data(imem_wr_data)
  );

  host_data_memory #(.DEPTH(4096)) u_hmem (
    .clk(clk), .rst_n(rst_n),
    .a_rd_en(hmem_a_rd_en), .a_rd_addr(hmem_a_rd_addr), .a_rd_data(hmem_a_rd_data),
    .a_wr_en(hmem_a_wr_en), .a_wr_addr(hmem_a_wr_addr), .a_wr_data(hmem_a_wr_data),
    .b_rd_en(hmem_b_rd_en), .b_rd_addr(hmem_b_rd_addr), .b_rd_data(hmem_b_rd_data),
    .b_wr_en(hmem_b_wr_en), .b_wr_addr(hmem_b_wr_addr), .b_wr_data(hmem_b_wr_data)
  );

  scratchpad_memory #(.DEPTH(4096)) u_spad (
    .clk(clk), .rst_n(rst_n),
    .a_rd_en(spad_a_rd_en), .a_rd_addr(spad_a_rd_addr), .a_rd_data(spad_a_rd_data),
    .a_wr_en(spad_a_wr_en), .a_wr_addr(spad_a_wr_addr), .a_wr_data(spad_a_wr_data),
    .b_rd_en(spad_b_rd_en), .b_rd_addr(spad_b_rd_addr), .b_rd_data(spad_b_rd_data),
    .b_wr_en(spad_b_wr_en), .b_wr_addr(spad_b_wr_addr), .b_wr_data(spad_b_wr_data)
  );

  weight_memory #(.DEPTH(65536)) u_weight (
    .clk(clk), .rst_n(rst_n),
    .rd_en(weight_rd_en), .rd_addr(weight_rd_addr), .rd_data(weight_rd_data),
    .wr_en(weight_wr_en), .wr_addr(weight_wr_addr), .wr_data(weight_wr_data)
  );

  bias_memory #(.DEPTH(65536)) u_bias (
    .clk(clk), .rst_n(rst_n),
    .rd_en(bias_rd_en), .rd_addr(bias_rd_addr), .rd_data(bias_rd_data),
    .wr_en(bias_wr_en), .wr_addr(bias_wr_addr), .wr_data(bias_wr_data)
  );

  output_memory #(.DEPTH(4096)) u_outmem (
    .clk(clk), .rst_n(rst_n),
    .wr_en(outmem_wr_en), .wr_addr(outmem_wr_addr), .wr_data(outmem_wr_data),
    .rd_en(outmem_rd_en), .rd_addr(outmem_rd_addr), .rd_data(outmem_rd_data)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 0; rst_n = 0; busy = 0;
    host_wr_en = 0; host_rd_en = 0; host_target_id = 0; host_addr = 0;
    host_wr_data32 = 0; host_wr_data8 = 0;

    hmem_a_rd_en = 0; hmem_a_wr_en = 0; hmem_a_rd_addr = 0; hmem_a_wr_addr = 0; hmem_a_wr_data = 0;
    spad_a_rd_en = 0; spad_a_wr_en = 0; spad_a_rd_addr = 0; spad_a_wr_addr = 0; spad_a_wr_data = 0;
    outmem_wr_en = 0; outmem_wr_addr = 0; outmem_wr_data = 0;

    #20 rst_n = 1;
    #10;

    $display("=== TEST 1: Write/Read Weight Memory (Target 3) ===");
    @(posedge clk);
    host_wr_en = 1;
    host_target_id = 3'd3;
    host_addr = 16'h00A0;
    host_wr_data8 = -8'sd42;
    @(posedge clk);
    host_wr_en = 0;

    // Read back
    host_rd_en = 1;
    host_addr = 16'h00A0;
    @(posedge clk);
    #1;
    assert(host_rd_data8 == -8'sd42) else $error("Weight mem readback failed: %d", host_rd_data8);
    host_rd_en = 0;

    $display("=== TEST 2: Write/Read Bias Memory (Target 4) ===");
    @(posedge clk);
    host_wr_en = 1;
    host_target_id = 3'd4;
    host_addr = 16'h0005;
    host_wr_data8 = 8'sd100;
    @(posedge clk);
    host_wr_en = 0;

    host_rd_en = 1;
    host_addr = 16'h0005;
    @(posedge clk);
    #1;
    assert(host_rd_data8 == 8'sd100) else $error("Bias mem readback failed: %d", host_rd_data8);
    host_rd_en = 0;

    $display("=== TEST 3: Output Memory Write & Host Readback (Target 5) ===");
    @(posedge clk);
    outmem_wr_en = 1;
    outmem_wr_addr = 12'd12;
    outmem_wr_data = 8'sd77;
    @(posedge clk);
    outmem_wr_en = 0;

    host_rd_en = 1;
    host_target_id = 3'd5;
    host_addr = 16'd12;
    @(posedge clk);
    #1;
    assert(host_rd_data8 == 8'sd77) else $error("Output mem host readback failed: %d", host_rd_data8);
    host_rd_en = 0;

    $display("=== ALL MEMORY SUBSYSTEM TESTS PASSED ===");
    $finish;
  end

endmodule
