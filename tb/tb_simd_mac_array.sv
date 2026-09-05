// =============================================================================
// tb_simd_mac_array.sv — Unit Testbench for SIMD MAC Array
// =============================================================================
// Uses hierarchical references to read products[] — workaround for the
// Icarus v12 bug where always_comb writing to unpacked-array output ports
// does not update the connected net in the instantiating module.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_simd_mac_array;

  localparam int W = 4;

  // Inputs driven as plain logic from initial block
  logic signed [7:0]  act_data   [W];
  logic signed [7:0]  weight_data[W];
  logic [W-1:0]       lanes_valid;
  logic signed [15:0] products   [W];

  simd_mac_array #(.SIMD_WIDTH(W)) u_dut (
    .act_data   (act_data),
    .weight_data(weight_data),
    .lanes_valid(lanes_valid),
    .products   (products)
  );

  integer fail_count;
  integer test_num;

  // Read via hierarchical reference to bypass Icarus net-propagation bug
  `define P0 $signed(u_dut.products[0])
  `define P1 $signed(u_dut.products[1])
  `define P2 $signed(u_dut.products[2])
  `define P3 $signed(u_dut.products[3])

  task automatic chk_hier(input int lane, input logic signed [15:0] exp);
    logic signed [15:0] got;
    case (lane)
      0: got = u_dut.products[0];
      1: got = u_dut.products[1];
      2: got = u_dut.products[2];
      3: got = u_dut.products[3];
    endcase
    if (got !== exp) begin
      $error("T%0d lane[%0d]: got=%0d  exp=%0d", test_num, lane, $signed(got), $signed(exp));
      fail_count++;
    end
  endtask

  task automatic show_hier();
    $display("  products = %0d  %0d  %0d  %0d", `P0, `P1, `P2, `P3);
  endtask

  initial begin
    fail_count = 0;
    test_num   = 0;
    for (int i = 0; i < W; i++) begin
      act_data[i]    = 8'sd0;
      weight_data[i] = 8'sd0;
    end
    lanes_valid = '0;
    #2;

    // =========================================================================
    // TEST 1: All lanes — positive × positive
    // =========================================================================
    test_num = 1;
    $display("=== TEST 1: All-lanes-valid, +x+ ===");
    act_data[0]=8'sd2;   weight_data[0]=8'sd3;
    act_data[1]=8'sd10;  weight_data[1]=8'sd5;
    act_data[2]=8'sd1;   weight_data[2]=8'sd1;
    act_data[3]=8'sd127; weight_data[3]=8'sd1;
    lanes_valid = 4'b1111;
    #1; show_hier();
    chk_hier(0, 16'sd6);   chk_hier(1, 16'sd50);
    chk_hier(2, 16'sd1);   chk_hier(3, 16'sd127);

    // =========================================================================
    // TEST 2: Negative activations
    // =========================================================================
    test_num = 2;
    $display("=== TEST 2: Negative activations ===");
    act_data[0]=-8'sd4;   weight_data[0]= 8'sd3;
    act_data[1]= 8'sd4;   weight_data[1]=-8'sd3;
    act_data[2]=-8'sd4;   weight_data[2]=-8'sd3;
    act_data[3]=-8'sd128; weight_data[3]= 8'sd1;
    lanes_valid = 4'b1111;
    #1; show_hier();
    chk_hier(0, -16'sd12); chk_hier(1, -16'sd12);
    chk_hier(2,  16'sd12); chk_hier(3, -16'sd128);

    // =========================================================================
    // TEST 3: Max-magnitude
    // =========================================================================
    test_num = 3;
    $display("=== TEST 3: Max-magnitude ===");
    act_data[0]=-8'sd128; weight_data[0]=-8'sd128;
    act_data[1]= 8'sd127; weight_data[1]= 8'sd127;
    act_data[2]= 8'sd0;   weight_data[2]= 8'sd0;
    act_data[3]= 8'sd1;   weight_data[3]= 8'sd1;
    lanes_valid = 4'b1111;
    #1; show_hier();
    chk_hier(0, 16'sd16384); chk_hier(1, 16'sd16129);
    chk_hier(2, 16'sd0);     chk_hier(3, 16'sd1);

    // =========================================================================
    // TEST 4: Partial mask 0101
    // =========================================================================
    test_num = 4;
    $display("=== TEST 4: Partial lane mask (0101) ===");
    act_data[0]=8'sd5; weight_data[0]=8'sd6;
    act_data[1]=8'sd9; weight_data[1]=8'sd9;
    act_data[2]=8'sd3; weight_data[2]=8'sd4;
    act_data[3]=8'sd7; weight_data[3]=8'sd7;
    lanes_valid = 4'b0101;
    #1; show_hier();
    chk_hier(0, 16'sd30); chk_hier(1, 16'sd0);
    chk_hier(2, 16'sd12); chk_hier(3, 16'sd0);

    // =========================================================================
    // TEST 5: All disabled
    // =========================================================================
    test_num = 5;
    $display("=== TEST 5: All lanes disabled ===");
    act_data[0]=8'sd99; weight_data[0]=8'sd99;
    act_data[1]=8'sd99; weight_data[1]=8'sd99;
    act_data[2]=8'sd99; weight_data[2]=8'sd99;
    act_data[3]=8'sd99; weight_data[3]=8'sd99;
    lanes_valid = 4'b0000;
    #1; show_hier();
    chk_hier(0, 16'sd0); chk_hier(1, 16'sd0);
    chk_hier(2, 16'sd0); chk_hier(3, 16'sd0);

    // =========================================================================
    // TEST 6: Single lane (lane 2 only, mask=0100)
    // =========================================================================
    test_num = 6;
    $display("=== TEST 6: Single lane (lane 2 only, mask=0100) ===");
    act_data[0]=8'sd1; weight_data[0]=8'sd5;
    act_data[1]=8'sd2; weight_data[1]=8'sd6;
    act_data[2]=8'sd10;weight_data[2]=8'sd7;
    act_data[3]=8'sd4; weight_data[3]=8'sd8;
    lanes_valid = 4'b0100;
    #1; show_hier();
    chk_hier(0, 16'sd0);  chk_hier(1, 16'sd0);
    chk_hier(2, 16'sd70); chk_hier(3, 16'sd0);

    // =========================================================================
    if (fail_count == 0)
      $display("=== ALL SIMD MAC ARRAY TESTS PASSED ===");
    else
      $display("!!! %0d FAILURE(S) DETECTED !!!", fail_count);
    $finish;
  end

endmodule
