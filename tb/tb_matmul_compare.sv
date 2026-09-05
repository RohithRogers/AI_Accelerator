`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_matmul_compare;
  localparam int W    = 8;
  localparam int AW   = 32;
  localparam int ILEN = 8;
  localparam int OLEN = 4;

  logic        clk = 0, rst_n = 0, start = 0;
  logic [15:0] input_addr, weight_addr, output_addr, bias_addr;
  logic [15:0] input_len  = ILEN;
  logic [15:0] output_len = OLEN;
  logic [7:0]  activation = 8'h00;  // ACT_NONE
  logic [31:0] requant_m0     = 32'h0000_0001;
  logic signed [7:0] requant_shift = -8'sd31;
  logic        busy, done, error;

  logic [W*16-1:0]        spad_rd_addr_bus;
  logic signed [W*8-1:0]  spad_rd_data_bus;
  logic                   spad_wr_en;
  logic [15:0]            spad_wr_addr;
  logic signed [7:0]      spad_wr_data;
  logic [W*16-1:0]        weight_rd_addr_bus;
  logic signed [W*8-1:0]  weight_rd_data_bus;
  logic [15:0]            bias_rd_addr;
  logic signed [7:0]      bias_rd_data;

  dense_engine #(.SIMD_WIDTH(W),.ACC_WIDTH(AW),.REQUANT_SHIFT(0),.RELU6_MAX(127)) u_dut (
    .clk(clk),.rst_n(rst_n),.start(start),
    .input_addr(input_addr),.weight_addr(weight_addr),
    .output_addr(output_addr),.bias_addr(bias_addr),
    .input_len(input_len),.output_len(output_len),
    .activation(activation),
    .requant_m0(requant_m0),.requant_shift(requant_shift),
    .busy(busy),.done(done),.error(error),
    .spad_rd_addr_bus(spad_rd_addr_bus),.spad_rd_data_bus(spad_rd_data_bus),
    .spad_wr_en(spad_wr_en),.spad_wr_addr(spad_wr_addr),.spad_wr_data(spad_wr_data),
    .weight_rd_addr_bus(weight_rd_addr_bus),.weight_rd_data_bus(weight_rd_data_bus),
    .bias_rd_addr(bias_rd_addr),.bias_rd_data(bias_rd_data)
  );

  logic signed [7:0] spad_mem   [0:4095];
  logic signed [7:0] weight_mem [0:4095];
  logic signed [7:0] bias_mem   [0:255];

  always_comb begin
    for (int i = 0; i < W; i++) begin
      spad_rd_data_bus  [i*8 +: 8] = spad_mem  [spad_rd_addr_bus  [i*16 +: 16]];
      weight_rd_data_bus[i*8 +: 8] = weight_mem[weight_rd_addr_bus[i*16 +: 16]];
    end
  end
  always_ff @(posedge clk) if (spad_wr_en) spad_mem[spad_wr_addr] <= spad_wr_data;
  assign bias_rd_data = bias_mem[bias_rd_addr];

  always #5 clk = ~clk;

  task automatic wait_done(input int timeout_cycles);
    int t = 0;
    @(posedge clk);
    while (!done && t < timeout_cycles) begin @(posedge clk); t++; end
    if (!done) begin $error("TIMEOUT"); $finish; end
  endtask

  integer fail_count;
  initial begin
    fail_count = 0;
    for (int i = 0; i < 4096; i++) spad_mem[i]   = 8'sd0;
    for (int i = 0; i < 4096; i++) weight_mem[i]  = 8'sd0;
    for (int i = 0; i < 256;  i++) bias_mem[i]    = 8'sd0;

    // -- Load input activations -----------------------------------------------
    spad_mem[0] = -8'sd36;
    spad_mem[1] = -8'sd58;
    spad_mem[2] = 8'sd6;
    spad_mem[3] = -8'sd2;
    spad_mem[4] = -8'sd7;
    spad_mem[5] = -8'sd29;
    spad_mem[6] = -8'sd38;
    spad_mem[7] = -8'sd42;

    // -- Load weight matrix (row-major) ---------------------------------------
    weight_mem[0] = 8'sd22;
    weight_mem[1] = -8'sd28;
    weight_mem[2] = -8'sd29;
    weight_mem[3] = -8'sd21;
    weight_mem[4] = -8'sd5;
    weight_mem[5] = -8'sd3;
    weight_mem[6] = 8'sd32;
    weight_mem[7] = -8'sd29;
    weight_mem[8] = -8'sd7;
    weight_mem[9] = 8'sd21;
    weight_mem[10] = -8'sd4;
    weight_mem[11] = 8'sd25;
    weight_mem[12] = 8'sd3;
    weight_mem[13] = -8'sd32;
    weight_mem[14] = -8'sd12;
    weight_mem[15] = 8'sd22;
    weight_mem[16] = 8'sd11;
    weight_mem[17] = 8'sd3;
    weight_mem[18] = -8'sd13;
    weight_mem[19] = -8'sd5;
    weight_mem[20] = 8'sd11;
    weight_mem[21] = -8'sd19;
    weight_mem[22] = -8'sd21;
    weight_mem[23] = 8'sd16;
    weight_mem[24] = -8'sd20;
    weight_mem[25] = 8'sd13;
    weight_mem[26] = 8'sd12;
    weight_mem[27] = 8'sd1;
    weight_mem[28] = -8'sd27;
    weight_mem[29] = 8'sd26;
    weight_mem[30] = -8'sd17;
    weight_mem[31] = 8'sd16;

    // -- Load biases ----------------------------------------------------------
    bias_mem[0] = -8'sd8;
    bias_mem[1] = 8'sd7;
    bias_mem[2] = -8'sd1;
    bias_mem[3] = 8'sd10;

    #20 rst_n = 1;
    @(posedge clk); #1;

    // -- Fire the engine ------------------------------------------------------
    input_addr  = 16'h0000;
    weight_addr = 16'h0000;
    output_addr = 16'h0200;
    bias_addr   = 16'h0000;

    @(posedge clk); start = 1; #1;
    @(posedge clk); start = 0;
    wait_done(1000);
    @(posedge clk);

    // -- Check outputs --------------------------------------------------------
    if (spad_mem[512] !== 8'sd127) begin
      $error("neuron[0]: got=%0d  exp=127", $signed(spad_mem[512]));
      fail_count++;
    end else
      $display("neuron[0] = %0d  OK", $signed(spad_mem[512]));
    if (spad_mem[513] !== -8'sd128) begin
      $error("neuron[1]: got=%0d  exp=-128", $signed(spad_mem[513]));
      fail_count++;
    end else
      $display("neuron[1] = %0d  OK", $signed(spad_mem[513]));
    if (spad_mem[514] !== -8'sd39) begin
      $error("neuron[2]: got=%0d  exp=-39", $signed(spad_mem[514]));
      fail_count++;
    end else
      $display("neuron[2] = %0d  OK", $signed(spad_mem[514]));
    if (spad_mem[515] !== -8'sd128) begin
      $error("neuron[3]: got=%0d  exp=-128", $signed(spad_mem[515]));
      fail_count++;
    end else
      $display("neuron[3] = %0d  OK", $signed(spad_mem[515]));

    if (fail_count == 0)
      $display("RTL_RESULT: ALL_PASS");
    else
      $display("RTL_RESULT: %0d_FAILURES", fail_count);
    $finish;
  end
endmodule
