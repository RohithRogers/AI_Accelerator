`timescale 1ns/1ps
// Isolated proof that packed lane buses work with the existing memory RTL.
module tb_packed_lane_memory;
  localparam int LANES = 4;
  logic clk = 0, rst_n = 0;
  logic [LANES*16-1:0] rd_addr_bus, addr_echo_bus;
  wire  [LANES*8-1:0]  rd_data_bus;
  logic [LANES*8-1:0]  adapter_data_bus;
  logic [LANES-1:0] wr_en;
  logic [LANES*8-1:0] wr_addr, wr_data;

  always #5 clk = ~clk;

  packed_lane_adapter #(.LANES(LANES)) adapter (
    .addr_bus(rd_addr_bus), .data_bus(rd_data_bus),
    .addr_echo_bus(addr_echo_bus), .data_echo_bus(adapter_data_bus)
  );

  generate
    for (genvar lane = 0; lane < LANES; lane++) begin : g_mem
      scratchpad_memory #(.DEPTH(16)) mem (
        .clk, .rst_n,
        .a_rd_en(1'b1), .a_rd_addr(addr_echo_bus[lane*16 +: 4]),
        .a_rd_data(rd_data_bus[lane*8 +: 8]),
        .a_wr_en(wr_en[lane]), .a_wr_addr(wr_addr[lane*8 +: 4]),
        .a_wr_data(wr_data[lane*8 +: 8]),
        .b_rd_en(1'b0), .b_rd_addr(4'd0), .b_rd_data(),
        .b_wr_en(1'b0), .b_wr_addr(4'd0), .b_wr_data(8'd0)
      );
    end
  endgenerate

  initial begin
    wr_en = '0; rd_addr_bus = '0;
    repeat (2) @(posedge clk); rst_n = 1;
    // Each memory replica gets the same four input bytes at addresses 0..3.
    for (int a = 0; a < 4; a++) begin
      wr_en = '1; wr_addr = {4{a[7:0]}}; wr_data = {4{8'(a + 1)}};
      @(posedge clk); wr_en = '0;
    end
    for (int lane = 0; lane < LANES; lane++) rd_addr_bus[lane*16 +: 16] = lane;
    @(posedge clk); #1;
    for (int lane = 0; lane < LANES; lane++)
      if ($signed(adapter_data_bus[lane*8 +: 8]) !== lane + 1)
        $fatal(1, "lane %0d got %0d expected %0d", lane,
               $signed(adapter_data_bus[lane*8 +: 8]), lane + 1);
    $display("PACKED_LANE_MEMORY_PASS data=[1,2,3,4]");
    $finish;
  end
endmodule
