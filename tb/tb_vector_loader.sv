// =============================================================================
// tb_vector_loader.sv — Unit Testbench for Vector Loader
// =============================================================================
// Uses hierarchical references (u_dut.spad_rd_addr[i], u_dut.weight_rd_addr[i])
// to bypass the Icarus v12 limitation with unpacked-array output ports.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_vector_loader;

  localparam int W = 4;

  logic        clk, rst_n;
  logic        start, next_chunk;
  logic [15:0] base_sp_addr, base_weight_addr, input_len;
  logic        busy, chunk_valid, last_chunk;
  logic [15:0] spad_rd_addr   [W];
  logic [15:0] weight_rd_addr [W];
  logic [W-1:0] lanes_valid;

  vector_loader #(.SIMD_WIDTH(W)) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (start),
    .next_chunk      (next_chunk),
    .base_sp_addr    (base_sp_addr),
    .base_weight_addr(base_weight_addr),
    .input_len       (input_len),
    .busy            (busy),
    .chunk_valid     (chunk_valid),
    .spad_rd_addr    (spad_rd_addr),
    .weight_rd_addr  (weight_rd_addr),
    .lanes_valid     (lanes_valid),
    .last_chunk      (last_chunk)
  );

  always #5 clk = ~clk;

  integer fail_count;

  // Read addresses via hierarchical reference (Icarus v12 unpacked-array fix)
  function automatic logic [15:0] get_spad(input int i);
    case (i)
      0: return u_dut.spad_rd_addr[0];
      1: return u_dut.spad_rd_addr[1];
      2: return u_dut.spad_rd_addr[2];
      3: return u_dut.spad_rd_addr[3];
      default: return 16'hxxxx;
    endcase
  endfunction

  function automatic logic [15:0] get_wt(input int i);
    case (i)
      0: return u_dut.weight_rd_addr[0];
      1: return u_dut.weight_rd_addr[1];
      2: return u_dut.weight_rd_addr[2];
      3: return u_dut.weight_rd_addr[3];
      default: return 16'hxxxx;
    endcase
  endfunction

  task automatic check_addrs(
    input int    chunk_idx,
    input logic [15:0] sp_base,
    input logic [15:0] wt_base,
    input int    offset
  );
    for (int i = 0; i < W; i++) begin
      if (get_spad(i) !== sp_base + offset + i) begin
        $error("Chunk%0d spad_rd_addr[%0d]: got %0d exp %0d",
          chunk_idx, i, get_spad(i), sp_base + offset + i);
        fail_count++;
      end
      if (get_wt(i) !== wt_base + offset + i) begin
        $error("Chunk%0d weight_rd_addr[%0d]: got %0d exp %0d",
          chunk_idx, i, get_wt(i), wt_base + offset + i);
        fail_count++;
      end
    end
  endtask

  initial begin
    clk   = 0; rst_n = 0;
    start = 0; next_chunk = 0;
    base_sp_addr = 0; base_weight_addr = 0; input_len = 0;
    fail_count = 0;
    #20 rst_n = 1;
    @(posedge clk); #1;

    // =========================================================================
    // TEST 1: input_len=8, two full chunks
    // =========================================================================
    $display("=== TEST 1: input_len=8, two full chunks ===");
    base_sp_addr     = 16'h0010;
    base_weight_addr = 16'h0100;
    input_len        = 16'd8;

    @(posedge clk); start = 1; #1;
    @(posedge clk); start = 0;

    @(posedge clk); #1;
    if (!chunk_valid) begin $error("T1: chunk_valid not set after start"); fail_count++; end
    if (!busy      ) begin $error("T1: busy not set after start");        fail_count++; end
    if ( last_chunk) begin $error("T1 chunk0: last_chunk should be 0");   fail_count++; end
    check_addrs(0, 16'h0010, 16'h0100, 0);
    if (lanes_valid !== 4'b1111) begin
      $error("T1 chunk0 lanes_valid: got %b exp 1111", lanes_valid); fail_count++;
    end

    @(posedge clk); next_chunk = 1; #1;
    @(posedge clk); next_chunk = 0;
    @(posedge clk); #1;
    if (!chunk_valid) begin $error("T1: chunk_valid not set for chunk1"); fail_count++; end
    if (!last_chunk ) begin $error("T1 chunk1: last_chunk should be 1"); fail_count++; end
    check_addrs(1, 16'h0010, 16'h0100, 4);
    if (lanes_valid !== 4'b1111) begin
      $error("T1 chunk1 lanes_valid: got %b exp 1111", lanes_valid); fail_count++;
    end

    @(posedge clk); next_chunk = 1; #1;
    @(posedge clk); next_chunk = 0;
    @(posedge clk); #1;
    if (busy      ) begin $error("T1: busy should clear after last chunk"); fail_count++; end
    if (chunk_valid) begin $error("T1: chunk_valid should clear when done"); fail_count++; end

    // =========================================================================
    // TEST 2: input_len=6, 1 full chunk then tail (2 lanes valid)
    // =========================================================================
    $display("=== TEST 2: input_len=6, tail masking ===");
    base_sp_addr     = 16'h0020;
    base_weight_addr = 16'h0200;
    input_len        = 16'd6;

    @(posedge clk); start = 1; #1;
    @(posedge clk); start = 0;
    @(posedge clk); #1;

    check_addrs(0, 16'h0020, 16'h0200, 0);
    if (lanes_valid !== 4'b1111) begin
      $error("T2 chunk0 lanes_valid: got %b exp 1111", lanes_valid); fail_count++;
    end
    if (last_chunk) begin $error("T2 chunk0: should not be last"); fail_count++; end

    @(posedge clk); next_chunk = 1; #1;
    @(posedge clk); next_chunk = 0;
    @(posedge clk); #1;

    check_addrs(1, 16'h0020, 16'h0200, 4);
    if (lanes_valid !== 4'b0011) begin
      $error("T2 chunk1 lanes_valid: got %b exp 0011", lanes_valid); fail_count++;
    end
    if (!last_chunk) begin $error("T2 chunk1: should be last"); fail_count++; end

    @(posedge clk); next_chunk = 1; #1;
    @(posedge clk); next_chunk = 0;
    @(posedge clk); #1;
    if (busy) begin $error("T2: busy should clear after tail"); fail_count++; end

    // =========================================================================
    // TEST 3: input_len=4, single chunk, immediately last
    // =========================================================================
    $display("=== TEST 3: input_len=4, single chunk ===");
    base_sp_addr     = 16'h0030;
    base_weight_addr = 16'h0300;
    input_len        = 16'd4;

    @(posedge clk); start = 1; #1;
    @(posedge clk); start = 0;
    @(posedge clk); #1;

    if (!last_chunk) begin $error("T3: last_chunk should be set immediately"); fail_count++; end
    if (lanes_valid !== 4'b1111) begin
      $error("T3 lanes_valid: got %b exp 1111", lanes_valid); fail_count++;
    end

    @(posedge clk); next_chunk = 1; #1;
    @(posedge clk); next_chunk = 0;
    @(posedge clk); #1;
    if (busy) begin $error("T3: busy should clear"); fail_count++; end

    // =========================================================================
    // TEST 4: input_len=1, only lane 0 valid, immediately last
    // =========================================================================
    $display("=== TEST 4: input_len=1, scalar ===");
    base_sp_addr     = 16'h0040;
    base_weight_addr = 16'h0400;
    input_len        = 16'd1;

    @(posedge clk); start = 1; #1;
    @(posedge clk); start = 0;
    @(posedge clk); #1;

    if (!last_chunk) begin $error("T4: last_chunk should be set"); fail_count++; end
    if (lanes_valid !== 4'b0001) begin
      $error("T4 lanes_valid: got %b exp 0001", lanes_valid); fail_count++;
    end
    if (get_spad(0) !== 16'h0040) begin
      $error("T4 spad[0]: got %0d exp 64", get_spad(0)); fail_count++;
    end

    @(posedge clk); next_chunk = 1; #1;
    @(posedge clk); next_chunk = 0;
    @(posedge clk); #1;
    if (busy) begin $error("T4: busy should clear"); fail_count++; end

    // =========================================================================
    if (fail_count == 0)
      $display("=== ALL VECTOR LOADER TESTS PASSED ===");
    else
      $display("!!! %0d FAILURE(S) DETECTED !!!", fail_count);

    $finish;
  end

endmodule
