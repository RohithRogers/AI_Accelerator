// =============================================================================
// tb_dense_engine.sv — Integration Testbench for Dense Engine
// =============================================================================
//
// Strategy:
//   This testbench instantiates dense_engine and wraps it with behavioural
//   single-port RAM models for scratchpad, weight memory, and bias memory.
//   It pre-loads known INT8 data, fires the engine, and checks every written
//   output byte against a golden value computed manually (and cross-checked
//   with the companion Python script  sw/compiler/test_dense_golden.py).
//
//   NOTE: dense_engine uses *packed* address/data buses.  The testbench
//   slices them with [lane*16 +: 16] / [lane*8 +: 8] notation.
//
// Test cases:
//   T1 — 1×4 dense: 1 output neuron, 4 inputs, no activation, SHIFT=0
//   T2 — 1×4 dense: 1 output neuron, RELU activation
//   T3 — 2×4 dense: 2 output neurons from separate weight rows
//   T4 — Tail-masking path: input_len=5 (not a multiple of SIMD_WIDTH=4)
//   T5 — Saturation: large accumulation clamps to −128
//
// Golden values:
//   Computed as:  out = sat8( bias + dot(act, weight) )  >> SHIFT=0
//   With RELU:    out = max(0, out)
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_dense_engine;

  // -------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------
  localparam int W     = 4;   // SIMD_WIDTH
  localparam int AW    = 32;  // ACC_WIDTH

  // -------------------------------------------------------------------------
  // DUT port wires
  // -------------------------------------------------------------------------
  logic        clk, rst_n;
  logic        start;
  logic [15:0] input_addr, weight_addr, output_addr, bias_addr;
  logic [15:0] input_len,  output_len;
  logic [7:0]  activation;
  logic [31:0] requant_m0;
  logic signed [7:0] requant_shift;
  logic        busy, done, error;

  // Packed memory buses (as dense_engine exposes them)
  logic [W*16-1:0]  spad_rd_addr_bus;
  logic signed [W*8-1:0]  spad_rd_data_bus;
  logic              spad_wr_en;
  logic [15:0]       spad_wr_addr;
  logic signed [7:0] spad_wr_data;

  logic [W*16-1:0]  weight_rd_addr_bus;
  logic signed [W*8-1:0]  weight_rd_data_bus;

  logic [15:0]       bias_rd_addr;
  logic signed [7:0] bias_rd_data;

  // -------------------------------------------------------------------------
  // DUT
  // -------------------------------------------------------------------------
  dense_engine #(
    .SIMD_WIDTH   (W),
    .ACC_WIDTH    (AW),
    .REQUANT_SHIFT(0),
    .RELU6_MAX    (127)
  ) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (start),
    .input_addr      (input_addr),
    .weight_addr     (weight_addr),
    .output_addr     (output_addr),
    .bias_addr       (bias_addr),
    .input_len       (input_len),
    .output_len      (output_len),
    .activation      (activation),
    .requant_m0      (requant_m0),
    .requant_shift   (requant_shift),
    .busy            (busy),
    .done            (done),
    .error           (error),
    .spad_rd_addr_bus   (spad_rd_addr_bus),
    .spad_rd_data_bus   (spad_rd_data_bus),
    .spad_wr_en         (spad_wr_en),
    .spad_wr_addr       (spad_wr_addr),
    .spad_wr_data       (spad_wr_data),
    .weight_rd_addr_bus (weight_rd_addr_bus),
    .weight_rd_data_bus (weight_rd_data_bus),
    .bias_rd_addr       (bias_rd_addr),
    .bias_rd_data       (bias_rd_data)
  );

  // -------------------------------------------------------------------------
  // Behavioural RAM models (combinational read)
  // -------------------------------------------------------------------------
  logic signed [7:0] spad_mem   [0:2047];
  logic signed [7:0] weight_mem [0:255];
  logic signed [7:0] bias_mem   [0:255];

  // Scratchpad: W simultaneous read lanes (packed bus) + 1 write port
  always_comb begin
    for (int i = 0; i < W; i++)
      spad_rd_data_bus[i*8 +: 8] = spad_mem[spad_rd_addr_bus[i*16 +: 16]];
  end

  always_ff @(posedge clk) begin
    if (spad_wr_en)
      spad_mem[spad_wr_addr] <= spad_wr_data;
  end

  // Weight memory: W simultaneous read lanes (packed bus)
  always_comb begin
    for (int i = 0; i < W; i++)
      weight_rd_data_bus[i*8 +: 8] = weight_mem[weight_rd_addr_bus[i*16 +: 16]];
  end

  // Bias memory: 1 read port
  assign bias_rd_data = bias_mem[bias_rd_addr];

  // -------------------------------------------------------------------------
  // Clock
  // -------------------------------------------------------------------------
  always #5 clk = ~clk;

  // -------------------------------------------------------------------------
  // Helper: wait for done (with timeout)
  // -------------------------------------------------------------------------
  task automatic wait_done(input string label, input int timeout_cycles);
    int t;
    t = 0;
    @(posedge clk);
    while (!done && t < timeout_cycles) begin
      @(posedge clk);
      t++;
    end
    if (!done)
      $error("[%s] TIMEOUT after %0d cycles — done never asserted", label, timeout_cycles);
    if (error)
      $error("[%s] Engine reported error=1", label);
  endtask

  // -------------------------------------------------------------------------
  // Helper: check scratchpad output byte
  // -------------------------------------------------------------------------
  integer fail_count;

  task automatic chk(
    input string label,
    input int    addr,
    input logic signed [7:0] exp
  );
    if (spad_mem[addr] !== exp) begin
      $error("[%s] spad[%0d]: got=%0d  exp=%0d",
        label, addr, $signed(spad_mem[addr]), $signed(exp));
      fail_count++;
    end else
      $display("[%s] spad[%0d] = %0d  OK", label, addr, $signed(spad_mem[addr]));
  endtask

  // -------------------------------------------------------------------------
  // Helper: fire one dense computation
  // -------------------------------------------------------------------------
  task automatic run_dense(
    input string  label,
    input logic [15:0] iaddr, waddr, oaddr, baddr,
    input logic [15:0] ilen,  olen,
    input logic [7:0]  act,
    input int          timeout
  );
    input_addr    = iaddr;
    weight_addr   = waddr;
    output_addr   = oaddr;
    bias_addr     = baddr;
    input_len     = ilen;
    output_len    = olen;
    activation    = act;
    requant_m0    = 32'h0000_0001; // mantissa=1, shift_n=-31 → total_shift=0 (identity)
    requant_shift = -8'sd31;

    @(posedge clk); start = 1; #1;
    @(posedge clk); start = 0;
    wait_done(label, timeout);
    @(posedge clk);               // one extra cycle for writeback register
    $display("[%s] finished.", label);
  endtask

  // =========================================================================
  // Main test sequence
  // =========================================================================
  initial begin
    clk        = 0;
    rst_n      = 0;
    start      = 0;
    fail_count = 0;
    requant_m0    = 32'h0000_0001; // mantissa=1, shift_n=-31 → total_shift=0 (identity)
    requant_shift = -8'sd31;

    // Zero all memories
    for (int i = 0; i < 2048; i++) spad_mem[i]   = 8'sd0;
    for (int i = 0; i < 256;  i++) weight_mem[i] = 8'sd0;
    for (int i = 0; i < 256;  i++) bias_mem[i]   = 8'sd0;

    #20 rst_n = 1;
    @(posedge clk); #1;

    // =========================================================================
    // TEST 1 — 1 output neuron, input_len=4, no activation
    // =========================================================================
    // spad[0..3]   = {1, 2, 3, 4}
    // weight[0..3] = {10, 20, 30, 40}
    // bias[0]      = 5
    // dot          = 1*10+2*20+3*30+4*40 = 300
    // biased       = 305  → sat8 → 127
    $display("=== TEST 1: 1 neuron, 4 inputs, NO-ACT (saturates to 127) ===");
    spad_mem[0] =  8'sd1;  spad_mem[1] =  8'sd2;
    spad_mem[2] =  8'sd3;  spad_mem[3] =  8'sd4;
    weight_mem[0] =  8'sd10; weight_mem[1] =  8'sd20;
    weight_mem[2] =  8'sd30; weight_mem[3] =  8'sd40;
    bias_mem[0]   =  8'sd5;

    run_dense("T1", 16'h0000, 16'h0000, 16'h0100, 16'h0000,
              16'd4, 16'd1, 8'h00 /*ACT_NONE*/, 100);
    @(posedge clk); #1;
    chk("T1", 16'h0100, 8'sd127);

    // =========================================================================
    // TEST 2 — Same weights, RELU activation; activations give negative dot
    // =========================================================================
    // spad[0..3] = {-3, -2, -1, 0}
    // dot = -3*10 + -2*20 + -1*30 + 0 = -100
    // biased = -95  → RELU → 0
    $display("=== TEST 2: 1 neuron, RELU, negative dot product ===");
    spad_mem[0] = -8'sd3; spad_mem[1] = -8'sd2;
    spad_mem[2] = -8'sd1; spad_mem[3] =  8'sd0;

    run_dense("T2", 16'h0000, 16'h0000, 16'h0200, 16'h0000,
              16'd4, 16'd1, 8'h01 /*ACT_RELU*/, 100);
    @(posedge clk); #1;
    chk("T2", 16'h0200, 8'sd0);

    // =========================================================================
    // TEST 3 — 2 output neurons, input_len=4
    // =========================================================================
    // Activations  spad[0..3] = {1, 2, 3, 4}
    // neuron 0: weight[16..19]={1,1,1,1} → dot=10, bias[4]=0 → out=10
    // neuron 1: weight[20..23]={2,2,2,2} → dot=20, bias[5]=0 → out=20
    $display("=== TEST 3: 2 neurons, 4 inputs each, no activation ===");
    spad_mem[0] = 8'sd1; spad_mem[1] = 8'sd2;
    spad_mem[2] = 8'sd3; spad_mem[3] = 8'sd4;
    weight_mem[16] = 8'sd1; weight_mem[17] = 8'sd1;
    weight_mem[18] = 8'sd1; weight_mem[19] = 8'sd1;
    weight_mem[20] = 8'sd2; weight_mem[21] = 8'sd2;
    weight_mem[22] = 8'sd2; weight_mem[23] = 8'sd2;
    bias_mem[4] = 8'sd0; bias_mem[5] = 8'sd0;

    run_dense("T3", 16'h0000, 16'h0010, 16'h0300, 16'h0004,
              16'd4, 16'd2, 8'h00 /*ACT_NONE*/, 200);
    @(posedge clk); #1;
    chk("T3.neuron0", 16'h0300, 8'sd10);
    chk("T3.neuron1", 16'h0301, 8'sd20);

    // =========================================================================
    // TEST 4 — Tail masking: input_len=5 (chunk0=4 lanes, chunk1=1 active lane)
    // =========================================================================
    // spad[32..36]   = {3, 0, 0, 0, 5}
    // weight[32..36] = {2, 0, 0, 0, 4}
    // dot = 3*2 + 5*4 = 26,  bias[8]=1 → out=27
    $display("=== TEST 4: input_len=5, tail masking (1 tail lane) ===");
    spad_mem[32] = 8'sd3;  spad_mem[33] = 8'sd0;
    spad_mem[34] = 8'sd0;  spad_mem[35] = 8'sd0;
    spad_mem[36] = 8'sd5;
    weight_mem[32] = 8'sd2; weight_mem[33] = 8'sd0;
    weight_mem[34] = 8'sd0; weight_mem[35] = 8'sd0;
    weight_mem[36] = 8'sd4;
    bias_mem[8] = 8'sd1;

    run_dense("T4", 16'h0020, 16'h0020, 16'h0400, 16'h0008,
              16'd5, 16'd1, 8'h00 /*ACT_NONE*/, 200);
    @(posedge clk); #1;
    chk("T4", 16'h0400, 8'sd27);

    // =========================================================================
    // TEST 5 — Saturation negative: clamps to -128
    // =========================================================================
    // spad[48..51] = {-128,-128,-128,-128}, weight[48..51]={127,127,127,127}
    // dot = 4*(-128*127) = -65024,  bias[12]=-127 → -65151 → sat8 → -128
    $display("=== TEST 5: Saturation to -128 ===");
    spad_mem[48] = -8'sd128; spad_mem[49] = -8'sd128;
    spad_mem[50] = -8'sd128; spad_mem[51] = -8'sd128;
    weight_mem[48] =  8'sd127; weight_mem[49] =  8'sd127;
    weight_mem[50] =  8'sd127; weight_mem[51] =  8'sd127;
    bias_mem[12] = -8'sd127;

    run_dense("T5", 16'h0030, 16'h0030, 16'h0500, 16'h000C,
              16'd4, 16'd1, 8'h00 /*ACT_NONE*/, 100);
    @(posedge clk); #1;
    chk("T5", 16'h0500, -8'sd128);

    // =========================================================================
    // TEST 6 — Error: input_len=0 should assert error immediately
    // =========================================================================
    $display("=== TEST 6: input_len=0, expect error flag ===");
    input_addr  = 16'h0000; weight_addr = 16'h0000;
    output_addr = 16'h0600; bias_addr   = 16'h0000;
    input_len   = 16'd0;    output_len  = 16'd1;
    activation  = 8'h00;

    // error is combinational from ST_IDLE: sample it the same cycle start is high
    @(posedge clk); start = 1;
    #1; // let combinational settle
    if (!error) begin
      $error("T6: error flag not set when input_len=0");
      fail_count++;
    end else
      $display("T6: error flag correctly raised.");
    @(posedge clk); start = 0;

    // =========================================================================
    // Summary
    // =========================================================================
    if (fail_count == 0)
      $display("=== ALL DENSE ENGINE TESTS PASSED ===");
    else
      $display("!!! %0d FAILURE(S) DETECTED !!!", fail_count);

    $finish;
  end

endmodule
