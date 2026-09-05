// =============================================================================
// tb_requantizer.sv — Unit Testbench for Requantizer
// =============================================================================
// Verifies arithmetic right-shift, INT8 saturation (both ends), and
// zero-passthrough. Tests several REQUANT_SHIFT values via DUT re-instantiation.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_requantizer;

  integer fail_count;

  // ---- DUT instances with different compile-time shifts --------------------
  // shift = 0 (identity except clamping)
  logic signed [31:0] in0;
  logic signed [7:0]  out0;
  requantizer #(.ACC_WIDTH(32), .REQUANT_SHIFT(0)) dut0 (.acc_in(in0), .val_out(out0));

  // shift = 4  (divide by 16)
  logic signed [31:0] in4;
  logic signed [7:0]  out4;
  requantizer #(.ACC_WIDTH(32), .REQUANT_SHIFT(4)) dut4 (.acc_in(in4), .val_out(out4));

  // shift = 8  (divide by 256)
  logic signed [31:0] in8;
  logic signed [7:0]  out8;
  requantizer #(.ACC_WIDTH(32), .REQUANT_SHIFT(8)) dut8 (.acc_in(in8), .val_out(out8));

  // ---- Helper ---------------------------------------------------------------
  task automatic chk8(
    input string      label,
    input logic signed [7:0] got,
    input logic signed [7:0] exp
  );
    if (got !== exp) begin
      $error("[%s] got=%0d exp=%0d", label, $signed(got), $signed(exp));
      fail_count++;
    end else
      $display("[%s] OK  → %0d", label, $signed(got));
  endtask

  initial begin
    fail_count = 0;

    // =========================================================================
    // TEST 1: REQUANT_SHIFT = 0 — pure saturation
    // =========================================================================
    $display("=== TEST 1: REQUANT_SHIFT=0 (saturation only) ===");

    // Value inside INT8 range
    in0 = 32'sd50;     #1; chk8("T1.a  50",      out0, 8'sd50);
    in0 = -32'sd50;    #1; chk8("T1.b -50",       out0, -8'sd50);
    in0 = 32'sd0;      #1; chk8("T1.c  0",        out0, 8'sd0);
    in0 = 32'sd127;    #1; chk8("T1.d  127",      out0, 8'sd127);
    in0 = -32'sd128;   #1; chk8("T1.e -128",      out0, -8'sd128);

    // Saturate high
    in0 = 32'sd128;    #1; chk8("T1.f  128→127",  out0, 8'sd127);
    in0 = 32'sd32767;  #1; chk8("T1.g  32767→127",out0, 8'sd127);
    in0 = 32'sd2147483647; #1; chk8("T1.h INT32MAX→127", out0, 8'sd127);

    // Saturate low
    in0 = -32'sd129;   #1; chk8("T1.i -129→-128", out0, -8'sd128);
    in0 = -32'sd2147483648; #1; chk8("T1.j INT32MIN→-128", out0, -8'sd128);

    // =========================================================================
    // TEST 2: REQUANT_SHIFT = 4 — arithmetic right-shift by 4
    // =========================================================================
    $display("=== TEST 2: REQUANT_SHIFT=4 ===");
    in4 = 32'sd1600;   #1; chk8("T2.a 1600>>4=100",   out4, 8'sd100);
    in4 = 32'sd16;     #1; chk8("T2.b 16>>4=1",        out4, 8'sd1);
    in4 = -32'sd16;    #1; chk8("T2.c -16>>4=-1",      out4, -8'sd1);
    in4 = 32'sd2032;   #1; chk8("T2.d 2032>>4=127",    out4, 8'sd127);
    in4 = 32'sd2048;   #1; chk8("T2.e 2048>>4→127 sat",out4, 8'sd127);
    in4 = -32'sd2048;  #1; chk8("T2.f -2048>>4=-128",  out4, -8'sd128);
    in4 = -32'sd2049;  #1; chk8("T2.g -2049>>4→-128 sat", out4, -8'sd128);
    // Arithmetic shift of negative — result should round toward -inf
    in4 = -32'sd1;     #1; chk8("T2.h -1>>>4=-1",      out4, -8'sd1);

    // =========================================================================
    // TEST 3: REQUANT_SHIFT = 8 — divide by 256
    // =========================================================================
    $display("=== TEST 3: REQUANT_SHIFT=8 ===");
    in8 = 32'sd25600;  #1; chk8("T3.a 25600>>8=100", out8, 8'sd100);
    in8 = 32'sd32512;  #1; chk8("T3.b 32512>>8=127", out8, 8'sd127);
    in8 = 32'sd32768;  #1; chk8("T3.c 32768>>8→127 sat", out8, 8'sd127);
    in8 = -32'sd32768; #1; chk8("T3.d -32768>>8=-128", out8, -8'sd128);
    in8 = 32'sd256;    #1; chk8("T3.e 256>>8=1",      out8, 8'sd1);
    in8 = 32'sd0;      #1; chk8("T3.f 0>>8=0",        out8, 8'sd0);

    // =========================================================================
    if (fail_count == 0)
      $display("=== ALL REQUANTIZER TESTS PASSED ===");
    else
      $display("!!! %0d FAILURE(S) DETECTED !!!", fail_count);

    $finish;
  end

endmodule
