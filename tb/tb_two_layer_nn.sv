`timescale 1ns/1ps
import tinyml_pkg::*;

// End-to-end dense test: 4 inputs -> 3 ReLU outputs -> 2 outputs.
// Four replicas model the compute engine's four simultaneous read lanes.
module tb_two_layer_nn;
  localparam int W = 4;
  logic clk = 0, rst_n = 0, start = 0;
  logic [15:0] input_addr, weight_addr, output_addr, bias_addr, input_len, output_len;
  logic [7:0] activation;
  logic [31:0] requant_m0; logic signed [7:0] requant_shift;
  logic busy, done, error;
  wire [W*16-1:0] spad_addr_bus, weight_addr_bus;
  wire signed [W*8-1:0] spad_data_bus, weight_data_bus;
  logic spad_wr_en; logic [15:0] spad_wr_addr; logic signed [7:0] spad_wr_data;
  logic [15:0] bias_rd_addr; logic signed [7:0] bias_rd_data;
  logic sp_b_wr_en; logic [7:0] sp_b_wr_addr, sp_b_rd_addr; logic signed [7:0] sp_b_wr_data;
  logic wt_wr_en; logic [7:0] wt_wr_addr; logic signed [7:0] wt_wr_data;
  logic bi_wr_en; logic [7:0] bi_wr_addr; logic signed [7:0] bi_wr_data;
  logic signed [7:0] observed [0:W-1];

  always #5 clk = ~clk;

  dense_engine #(.SIMD_WIDTH(W), .REQUANT_SHIFT(0)) dut (
    .clk, .rst_n, .start, .input_addr, .weight_addr, .output_addr, .bias_addr,
    .input_len, .output_len, .activation, .requant_m0, .requant_shift, .busy, .done, .error,
    .spad_rd_addr_bus(spad_addr_bus), .spad_rd_data_bus(spad_data_bus), .spad_wr_en, .spad_wr_addr,
    .spad_wr_data, .weight_rd_addr_bus(weight_addr_bus), .weight_rd_data_bus(weight_data_bus),
    .bias_rd_addr, .bias_rd_data
  );

  generate
    for (genvar lane = 0; lane < W; lane++) begin : g_mem
      scratchpad_memory #(.DEPTH(256)) spad (
        .clk, .rst_n, .a_rd_en(1'b1), .a_rd_addr(spad_addr_bus[lane*16 +: 8]),
        .a_rd_data(spad_data_bus[lane*8 +: 8]), .a_wr_en(spad_wr_en), .a_wr_addr(spad_wr_addr[7:0]),
        .a_wr_data(spad_wr_data), .b_rd_en(1'b1), .b_rd_addr(sp_b_rd_addr),
        .b_rd_data(observed[lane]), .b_wr_en(sp_b_wr_en), .b_wr_addr(sp_b_wr_addr),
        .b_wr_data(sp_b_wr_data)
      );
      weight_memory #(.DEPTH(256)) weights (
        .clk, .rst_n, .rd_en(1'b1), .rd_addr(weight_addr_bus[lane*16 +: 8]),
        .rd_data(weight_data_bus[lane*8 +: 8]), .wr_en(wt_wr_en), .wr_addr(wt_wr_addr), .wr_data(wt_wr_data)
      );
    end
  endgenerate

  bias_memory #(.DEPTH(256)) biases (
    .clk, .rst_n, .rd_en(1'b1), .rd_addr(bias_rd_addr[7:0]), .rd_data(bias_rd_data),
    .wr_en(bi_wr_en), .wr_addr(bi_wr_addr), .wr_data(bi_wr_data)
  );

  task automatic preload_spad(input [7:0] addr, input logic signed [7:0] value);
    @(negedge clk); sp_b_wr_addr = addr; sp_b_wr_data = value; sp_b_wr_en = 1;
    @(posedge clk); @(negedge clk); sp_b_wr_en = 0;
  endtask
  task automatic preload_weight(input [7:0] addr, input logic signed [7:0] value);
    @(negedge clk); wt_wr_addr = addr; wt_wr_data = value; wt_wr_en = 1;
    @(posedge clk); @(negedge clk); wt_wr_en = 0;
  endtask
  task automatic preload_bias(input [7:0] addr, input logic signed [7:0] value);
    @(negedge clk); bi_wr_addr = addr; bi_wr_data = value; bi_wr_en = 1;
    @(posedge clk); @(negedge clk); bi_wr_en = 0;
  endtask
  task automatic run_dense(input [15:0] ia, wa, oa, ba, il, ol, input [7:0] act);
    input_addr=ia; weight_addr=wa; output_addr=oa; bias_addr=ba; input_len=il; output_len=ol; activation=act;
    requant_m0=32'h4000_0000; requant_shift=-8'sd1;
    @(negedge clk); start=1; @(negedge clk); start=0;
    while (!done) @(posedge clk);
    if (error) $fatal(1, "dense engine error");
    @(posedge clk); // registered writeback commits
  endtask

  initial begin
    sp_b_wr_en=0; wt_wr_en=0; bi_wr_en=0;
    repeat (2) @(posedge clk); rst_n=1;
    // x = [1,-2,3,2]
    preload_spad(0,1); preload_spad(1,-2); preload_spad(2,3); preload_spad(3,2);
    // W1 rows: [1,2,-1,1], [-2,1,1,0], [3,-1,0,2]
    preload_weight(0,1); preload_weight(1,2); preload_weight(2,-1); preload_weight(3,1);
    preload_weight(4,-2); preload_weight(5,1); preload_weight(6,1); preload_weight(7,0);
    preload_weight(8,3); preload_weight(9,-1); preload_weight(10,0); preload_weight(11,2);
    preload_bias(0,5); preload_bias(1,1); preload_bias(2,-2);
    // W2 rows: [2,-1,3], [-1,4,-2]
    preload_weight(32,2); preload_weight(33,-1); preload_weight(34,3);
    preload_weight(35,-1); preload_weight(36,4); preload_weight(37,-2);
    preload_bias(16,-3); preload_bias(17,5);

    run_dense(0, 0, 64, 0, 4, 3, ACT_RELU);
    run_dense(64, 32, 80, 16, 3, 2, ACT_NONE);
    // Port-B reads are registered; sample separately to avoid read/write races.
    @(negedge clk); sp_b_rd_addr = 8'd80; @(posedge clk); #1;
    if (observed[0] !== 8'sd20) $fatal(1, "y0 got=%0d expected=20", observed[0]);
    @(negedge clk); sp_b_rd_addr = 8'd81; @(posedge clk); #1;
    if (observed[0] !== -8'sd10) $fatal(1, "y1 got=%0d expected=-10", observed[0]);
    $display("TWO_LAYER_NN_PASS y=[%0d,%0d]", 20, -10);
    $finish;
  end
endmodule
