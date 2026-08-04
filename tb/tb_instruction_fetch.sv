// =============================================================================
// tb_instruction_fetch.sv — Testbench for Instruction Fetch and Program Counter
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_instruction_fetch;

  logic clk;
  logic rst_n;

  // Program counter signals
  logic        pc_write;
  logic [15:0] pc_next;
  logic [15:0] pc_out;

  // Instruction fetch signals
  logic        fetch_start;
  logic        instr_valid;
  logic        fetch_error;
  logic [31:0] w0, w1, w2, w3;
  logic [15:0] next_pc;

  // Instruction memory signals
  logic        imem_rd_en, imem_wr_en;
  logic [15:0] imem_rd_addr;
  logic [9:0]  imem_wr_addr;
  logic [31:0] imem_rd_data, imem_wr_data;

  // Instruction decoder output
  decoded_instr_t instr_out;

  // Modules instantiation
  program_counter u_pc (
    .clk(clk),
    .rst_n(rst_n),
    .pc_write(pc_write),
    .pc_next(pc_next),
    .pc_out(pc_out)
  );

  instruction_memory #(.DEPTH(1024)) u_imem (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(imem_rd_en),
    .rd_addr(imem_rd_addr[9:0]),
    .rd_data(imem_rd_data),
    .wr_en(imem_wr_en),
    .wr_addr(imem_wr_addr),
    .wr_data(imem_wr_data)
  );

  instruction_fetch u_fetch (
    .clk(clk),
    .rst_n(rst_n),
    .fetch_start(fetch_start),
    .instr_valid(instr_valid),
    .fetch_error(fetch_error),
    .pc_in(pc_out),
    .w0(w0), .w1(w1), .w2(w2), .w3(w3),
    .next_pc(next_pc),
    .imem_rd_en(imem_rd_en),
    .imem_rd_addr(imem_rd_addr),
    .imem_rd_data(imem_rd_data)
  );

  instruction_decoder u_decoder (
    .w0(w0), .w1(w1), .w2(w2), .w3(w3),
    .instr_out(instr_out)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 0; rst_n = 0; pc_write = 0; pc_next = 0; fetch_start = 0;
    imem_wr_en = 0; imem_wr_addr = 0; imem_wr_data = 0;

    #20 rst_n = 1;
    #10;

    $display("=== Preloading Test Program into IMEM ===");
    // Word 0..2: LOAD instruction (3 words)
    // W0: OP_LOAD (0x01), flags=0x00, act=NONE -> 32'h01000000
    // W1: mem_addr=0x0100, sp_addr=0x0010 -> 32'h01000010
    // W2: length=0x0020, res=0x0000 -> 32'h00200000
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd0; imem_wr_data = 32'h01000000;
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd1; imem_wr_data = 32'h01000010;
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd2; imem_wr_data = 32'h00200000;

    // Word 3..6: DENSE instruction (4 words)
    // W0: OP_DENSE (0x03), flags=0x00, act=RELU (0x01) -> 32'h03000100
    // W1: input_addr=0x0010, weight_addr=0x0080 -> 32'h00100080
    // W2: output_addr=0x0040, bias_addr=0x0004 -> 32'h00400004
    // W3: input_len=0x0008, output_len=0x0004 -> 32'h00080004
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd3; imem_wr_data = 32'h03000100;
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd4; imem_wr_data = 32'h00100080;
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd5; imem_wr_data = 32'h00400004;
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd6; imem_wr_data = 32'h00080004;

    // Word 7: END instruction (1 word)
    // W0: OP_END (0x05) -> 32'h05000000
    @(posedge clk); imem_wr_en = 1; imem_wr_addr = 10'd7; imem_wr_data = 32'h05000000;
    @(posedge clk); imem_wr_en = 0;

    #20;

    $display("=== TEST 1: Fetch LOAD Instruction at PC=0 ===");
    assert(pc_out == 16'd0) else $error("PC should be 0");
    @(posedge clk); fetch_start = 1;
    @(posedge clk); fetch_start = 0;

    @(posedge instr_valid);
    #1;
    assert(instr_out.opcode == OP_LOAD) else $error("Fetched instruction should be OP_LOAD");
    assert(instr_out.mem_addr == 16'h0100) else $error("mem_addr mismatch");
    assert(instr_out.sp_addr == 16'h0010) else $error("sp_addr mismatch");
    assert(instr_out.length == 16'h0020) else $error("length mismatch");
    assert(next_pc == 16'd3) else $error("next_pc should be 3 for 3-word LOAD");

    $display("=== Retiring LOAD Instruction (Advance PC to 3) ===");
    @(posedge clk);
    pc_write = 1;
    pc_next = next_pc;
    @(posedge clk);
    pc_write = 0;
    #1;
    assert(pc_out == 16'd3) else $error("PC should be advanced to 3");

    $display("=== TEST 2: Fetch DENSE Instruction at PC=3 ===");
    @(posedge clk); fetch_start = 1;
    @(posedge clk); fetch_start = 0;

    @(posedge instr_valid);
    #1;
    assert(instr_out.opcode == OP_DENSE) else $error("Fetched instruction should be OP_DENSE");
    assert(instr_out.activation == ACT_RELU) else $error("activation mismatch");
    assert(instr_out.input_addr == 16'h0010) else $error("input_addr mismatch");
    assert(instr_out.weight_addr == 16'h0080) else $error("weight_addr mismatch");
    assert(instr_out.output_addr == 16'h0040) else $error("output_addr mismatch");
    assert(instr_out.bias_addr == 16'h0004) else $error("bias_addr mismatch");
    assert(instr_out.input_len == 16'h0008) else $error("input_len mismatch");
    assert(instr_out.output_len == 16'h0004) else $error("output_len mismatch");
    assert(next_pc == 16'd7) else $error("next_pc should be 7 for 4-word DENSE");

    $display("=== Retiring DENSE Instruction (Advance PC to 7) ===");
    @(posedge clk);
    pc_write = 1;
    pc_next = next_pc;
    @(posedge clk);
    pc_write = 0;
    #1;
    assert(pc_out == 16'd7) else $error("PC should be advanced to 7");

    $display("=== TEST 3: Fetch END Instruction at PC=7 ===");
    @(posedge clk); fetch_start = 1;
    @(posedge clk); fetch_start = 0;

    @(posedge instr_valid);
    #1;
    assert(instr_out.opcode == OP_END) else $error("Fetched instruction should be OP_END");
    assert(next_pc == 16'd8) else $error("next_pc should be 8 for 1-word END");

    $display("=== ALL FETCH & PC TESTS PASSED ===");
    $finish;
  end

endmodule
