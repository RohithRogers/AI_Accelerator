// =============================================================================
// tb_instruction_decoder.sv — Testbench for Instruction Decoder and Memory
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_instruction_decoder;

  logic clk;
  logic rst_n;

  // IMEM ports
  logic imem_rd_en;
  logic [9:0] imem_rd_addr;
  logic [31:0] imem_rd_data;
  logic imem_wr_en;
  logic [9:0] imem_wr_addr;
  logic [31:0] imem_wr_data;

  // Decoder ports
  logic [31:0] w0, w1, w2, w3;
  decoded_instr_t instr_out;

  // Instantiations
  instruction_memory #(
    .DEPTH(1024)
  ) u_imem (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(imem_rd_en),
    .rd_addr(imem_rd_addr),
    .rd_data(imem_rd_data),
    .wr_en(imem_wr_en),
    .wr_addr(imem_wr_addr),
    .wr_data(imem_wr_data)
  );

  instruction_decoder u_decoder (
    .w0(w0),
    .w1(w1),
    .w2(w2),
    .w3(w3),
    .instr_out(instr_out)
  );

  // Clock generation
  always #5 clk = ~clk;

  initial begin
    clk = 0;
    rst_n = 0;
    imem_rd_en = 0;
    imem_rd_addr = 0;
    imem_wr_en = 0;
    imem_wr_addr = 0;
    imem_wr_data = 0;

    w0 = 0; w1 = 0; w2 = 0; w3 = 0;

    #20 rst_n = 1;
    #10;

    $display("=== TEST 1: NOP instruction decode ===");
    w0 = 32'h00000000;
    w1 = 0; w2 = 0; w3 = 0;
    #1;
    assert(instr_out.valid == 1'b1) else $error("NOP should be valid");
    assert(instr_out.opcode == OP_NOP) else $error("Opcode should be OP_NOP");

    $display("=== TEST 2: LOAD instruction decode ===");
    w0 = 32'h01000000;
    w1 = 32'h10000020;
    w2 = 32'h00400000;
    w3 = 32'h00000000;
    #1;
    assert(instr_out.valid == 1'b1) else $error("LOAD should be valid");
    assert(instr_out.opcode == OP_LOAD) else $error("Opcode should be OP_LOAD");
    assert(instr_out.mem_addr == 16'h1000) else $error("mem_addr mismatch");
    assert(instr_out.sp_addr == 16'h0020) else $error("sp_addr mismatch");
    assert(instr_out.length == 16'h0040) else $error("length mismatch");

    $display("=== TEST 3: STORE instruction decode ===");
    // Header: OP_STORE (0x02), flags=0x00, act=NONE (0x00) -> 32'h02000000
    // W1: sp_addr=0x0050, mem_addr=0x2000 -> 32'h00502000
    // W2: length=0x0010, res=0x0000 -> 32'h00100000
    w0 = 32'h02000000;
    w1 = 32'h00502000;
    w2 = 32'h00100000;
    w3 = 32'h00000000;
    #1;
    assert(instr_out.valid == 1'b1) else $error("STORE should be valid");
    assert(instr_out.opcode == OP_STORE) else $error("Opcode should be OP_STORE");
    assert(instr_out.sp_addr == 16'h0050) else $error("STORE sp_addr mismatch");
    assert(instr_out.mem_addr == 16'h2000) else $error("STORE mem_addr mismatch");
    assert(instr_out.length == 16'h0010) else $error("STORE length mismatch");

    $display("=== TEST 4: DENSE instruction decode ===");
    w0 = 32'h03000100;
    w1 = 32'h00100100;
    w2 = 32'h00800008;
    w3 = 32'h00040008;
    #1;
    assert(instr_out.valid == 1'b1) else $error("DENSE should be valid");
    assert(instr_out.opcode == OP_DENSE) else $error("Opcode should be OP_DENSE");
    assert(instr_out.activation == ACT_RELU) else $error("activation mismatch");
    assert(instr_out.input_addr == 16'h0010) else $error("input_addr mismatch");
    assert(instr_out.weight_addr == 16'h0100) else $error("weight_addr mismatch");
    assert(instr_out.output_addr == 16'h0080) else $error("output_addr mismatch");
    assert(instr_out.bias_addr == 16'h0008) else $error("bias_addr mismatch");
    assert(instr_out.input_len == 16'h0004) else $error("input_len mismatch");
    assert(instr_out.output_len == 16'h0008) else $error("output_len mismatch");

    $display("=== TEST 5: ACT instruction decode ===");
    // Header: OP_ACT (0x04), flags=0x00, act=RELU6 (0x02) -> 32'h04000200
    // W1: addr=0x0030, length=0x0010 -> 32'h00300010
    w0 = 32'h04000200;
    w1 = 32'h00300010;
    w2 = 32'h00000000;
    w3 = 32'h00000000;
    #1;
    assert(instr_out.valid == 1'b1) else $error("ACT should be valid");
    assert(instr_out.opcode == OP_ACT) else $error("Opcode should be OP_ACT");
    assert(instr_out.activation == ACT_RELU6) else $error("ACT activation mismatch");
    assert(instr_out.sp_addr == 16'h0030) else $error("ACT sp_addr mismatch");
    assert(instr_out.length == 16'h0010) else $error("ACT length mismatch");

    $display("=== TEST 6: Decode Error (Illegal opcode) ===");
    w0 = 32'hFF000000;
    #1;
    assert(instr_out.valid == 1'b0) else $error("Illegal opcode should fail valid");
    assert(instr_out.decode_error == 1'b1) else $error("Illegal opcode should raise decode_error");

    $display("=== TEST 7: Instruction Memory Write and 1-Cycle Read ===");
    @(posedge clk);
    imem_wr_en = 1;
    imem_wr_addr = 10'd5;
    imem_wr_data = 32'hDEADBEEF;
    @(posedge clk);
    imem_wr_en = 0;

    imem_rd_en = 1;
    imem_rd_addr = 10'd5;
    @(posedge clk);
    #1;
    assert(imem_rd_data == 32'hDEADBEEF) else $error("IMEM readback mismatch");

    $display("=== ALL DECODER & IMEM TESTS PASSED ===");
    $finish;
  end

endmodule
