// =============================================================================
// tb_conv_engine.sv — Convolution integration test for CONV_CFG + CONV flow
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_conv_engine;

  logic clk;
  logic rst_n;

  logic        start;
  logic        busy;
  logic        done;
  logic        error;
  logic [7:0]  error_code;

  logic        host_wr_en;
  logic        host_rd_en;
  logic [2:0]  host_target_id;
  logic [15:0] host_addr;
  logic [31:0] host_wr_data32;
  logic signed [7:0] host_wr_data8;
  logic [31:0] host_rd_data32;
  logic signed [7:0] host_rd_data8;

  integer fail_count;
  integer debug_cycles;

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (debug_cycles < 100) begin
      $display("t=%0t ctrl=%0d fetch=%0d dense=%0d im2col=%0d busy=%b", $time,
               dut.u_controller.current_state, dut.u_fetch.state,
               dut.u_dense_engine.state, dut.u_im2col.state, busy);
      $fflush();
      debug_cycles = debug_cycles + 1;
    end
  end

  tinyml_accelerator_top #(
    .IMEM_DEPTH_WORDS (1024),
    .SPAD_DEPTH       (4096),
    .PARAM_DEPTH      (1024),
    .OUT_DEPTH        (4096),
    .HMEM_DEPTH       (4096),
    .SIMD_WIDTH       (1),
    .ACC_WIDTH        (32),
    .REQUANT_SHIFT    (0),
    .RELU6_MAX        (127)
  ) dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (start),
    .busy            (busy),
    .done            (done),
    .error           (error),
    .error_code      (error_code),
    .host_wr_en      (host_wr_en),
    .host_rd_en      (host_rd_en),
    .host_target_id  (host_target_id),
    .host_addr       (host_addr),
    .host_wr_data32  (host_wr_data32),
    .host_wr_data8   (host_wr_data8),
    .host_rd_data32  (host_rd_data32),
    .host_rd_data8   (host_rd_data8)
  );

  task automatic write_imem(input [15:0] addr, input [31:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd0;
    host_addr      = addr;
    host_wr_data32 = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic write_spad(input [15:0] addr, input logic signed [7:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd2;
    host_addr      = addr;
    host_wr_data8  = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic write_weight(input [15:0] addr, input logic signed [7:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd3;
    host_addr      = addr;
    host_wr_data8  = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic write_bias(input [15:0] addr, input logic signed [7:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd4;
    host_addr      = addr;
    host_wr_data8  = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic read_spad(input [15:0] addr, output logic signed [7:0] data);
    @(negedge clk);
    host_rd_en     = 1'b1;
    host_target_id = 3'd2;
    host_addr      = addr;
    @(posedge clk);
    #1;
    data = host_rd_data8;
    @(negedge clk);
    host_rd_en     = 1'b0;
  endtask

  task automatic run_program(input int timeout_cycles);
    int cycles;
    cycles = 0;
    @(negedge clk);
    start = 1'b1;
    @(posedge clk);
    @(negedge clk);
    start = 1'b0;

    while (!done && !error && cycles < timeout_cycles) begin
      @(posedge clk);
      cycles++;
    end

    if (cycles >= timeout_cycles) begin
      $error("TIMEOUT after %0d cycles", timeout_cycles);
      fail_count++;
    end else if (error) begin
      $error("Execution ERRORED with code=0x%02x", error_code);
      fail_count++;
    end else begin
      $display("Program completed in %0d cycles.", cycles);
    end
  endtask

  initial begin
    logic signed [7:0] got;
    $display("tb_conv_engine: starting");
    $fflush();
    clk            = 0;
    rst_n          = 0;
    start          = 0;
    host_wr_en     = 0;
    host_rd_en     = 0;
    host_target_id = 0;
    host_addr      = 0;
    host_wr_data32 = 0;
    host_wr_data8  = 0;
    fail_count     = 0;
    debug_cycles   = 0;


    #20 rst_n = 1;
    #20;
    $display("tb_conv_engine: reset released");
    $fflush();

    // ---------------------------------------------------------------------
    // 4x4 input, 1 channel, two 3x3 filters, no pad, stride 1 => 2x2x2 output.
    // Input feature map is loaded into SPAD at address 0x0000.
    // Kernel weights are stored in weight memory at address 0x0100.
    // Final output is at 0x0200; im2col temporary patches begin at 0x0204.
    // Both ranges are inside the 4 KiB scratchpad (0x0000..0x0fff).
    // ---------------------------------------------------------------------
    $display("=== CONV TEST: 1x1 4x4 -> 2x2, 3x3 kernel, no padding ===");

    // 4x4 input image (flattened row-major): 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16
    write_spad(16'd0,  8'sd1);  write_spad(16'd1,  8'sd2);  write_spad(16'd2,  8'sd3);  write_spad(16'd3,  8'sd4);
    write_spad(16'd4,  8'sd5);  write_spad(16'd5,  8'sd6);  write_spad(16'd6,  8'sd7);  write_spad(16'd7,  8'sd8);
    write_spad(16'd8,  8'sd9);  write_spad(16'd9,  8'sd10); write_spad(16'd10, 8'sd11); write_spad(16'd11, 8'sd12);
    write_spad(16'd12, 8'sd13); write_spad(16'd13, 8'sd14); write_spad(16'd14, 8'sd15); write_spad(16'd15, 8'sd16);

    // Filter 0 = all ones.  Filter 1 selects the patch top-left value.
    for (int i = 0; i < 9; i++) write_weight(16'h0100 + i, 8'sd1);
    write_weight(16'h0109, 8'sd1);
    for (int i = 1; i < 9; i++) write_weight(16'h0109 + i, 8'sd0);

    // One bias per output channel, not per output pixel.
    write_bias(16'h0300, 8'sd0);
    write_bias(16'h0301, 8'sd1);

    // CONV_CFG: C_in=1, out_channels=2, H_in=4, W_in=4, K=3, stride=1, pad=0
    write_imem(16'd0, 32'h06000000);
    write_imem(16'd1, 32'h00010002);
    write_imem(16'd2, 32'h00040004);
    write_imem(16'd3, 32'h00030003);
    write_imem(16'd4, 32'h00010000);

    // CONV: input_addr=0x0000, weight_addr=0x0100, output_addr=0x0200, bias_addr=0x0300,
    //      out_h=2, out_w=2, activation=NONE, shift=-31, M0=1
    write_imem(16'd5, 32'h070000e1);
    write_imem(16'd6, 32'h00000100);
    write_imem(16'd7, 32'h02000300);
    write_imem(16'd8, 32'h00020002);
    write_imem(16'd9, 32'h00000001);

    // END
    write_imem(16'd10, 32'h05000000);

    $display("tb_conv_engine: program loaded");
    $fflush();

    run_program(2000);

    if (!error) begin
      // Output layout is spatial-major: [pixel0_ch0, pixel0_ch1, pixel1_ch0, ...].
      read_spad(16'h0200, got); $display("spad[0x0200] = %0d", $signed(got));
      if ($signed(got) !== 54) begin $error("Output[0] got=%0d expected=54", $signed(got)); fail_count++; end

      read_spad(16'h0201, got); $display("spad[0x0201] = %0d", $signed(got));
      if ($signed(got) !== 2) begin $error("Output[1] got=%0d expected=2", $signed(got)); fail_count++; end

      read_spad(16'h0202, got); $display("spad[0x0202] = %0d", $signed(got));
      if ($signed(got) !== 63) begin $error("Output[2] got=%0d expected=63", $signed(got)); fail_count++; end

      read_spad(16'h0203, got); $display("spad[0x0203] = %0d", $signed(got));
      if ($signed(got) !== 3) begin $error("Output[3] got=%0d expected=3", $signed(got)); fail_count++; end

      read_spad(16'h0204, got); if ($signed(got) !== 90) begin $error("Output[4] got=%0d expected=90", $signed(got)); fail_count++; end
      read_spad(16'h0205, got); if ($signed(got) !== 6)  begin $error("Output[5] got=%0d expected=6",  $signed(got)); fail_count++; end
      read_spad(16'h0206, got); if ($signed(got) !== 99) begin $error("Output[6] got=%0d expected=99", $signed(got)); fail_count++; end
      read_spad(16'h0207, got); if ($signed(got) !== 7)  begin $error("Output[7] got=%0d expected=7",  $signed(got)); fail_count++; end
    end

    if (fail_count == 0) begin
      $display("=== ALL CONV ENGINE TESTS PASSED ===");
    end else begin
      $display("!!! CONV TEST FAILURES = %0d !!!", fail_count);
    end

    $finish;
  end

endmodule
