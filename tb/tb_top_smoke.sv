`timescale 1ns/1ps

module tb_top_smoke;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic start = 1'b0;
  logic host_wr_en = 1'b0;
  logic host_rd_en = 1'b0;
  logic [2:0] host_target_id = '0;
  logic [15:0] host_addr = '0;
  logic [31:0] host_wr_data32 = '0;
  logic signed [7:0] host_wr_data8 = '0;
  logic busy, done, error;
  logic [7:0] error_code;
  logic [31:0] host_rd_data32;
  logic signed [7:0] host_rd_data8;

  always #5 clk = ~clk;

  tinyml_accelerator_top #(
    .IMEM_DEPTH_WORDS(16), .SPAD_DEPTH(64), .PARAM_DEPTH(64),
    .OUT_DEPTH(64), .HMEM_DEPTH(64), .SIMD_WIDTH(1)
  ) dut (.*);

  initial begin
    #20 rst_n = 1'b1;
    #20;
    $display("TOP SMOKE PASS at %0t", $time);
    $finish;
  end
endmodule
