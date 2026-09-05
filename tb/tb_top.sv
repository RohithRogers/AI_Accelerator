// =============================================================================
// tb_top.sv — End-to-End System-Level Testbench for TinyML Accelerator
//
// PURPOSE:
//   Validates the entire integrated TinyML accelerator:
//     - Host preloading of instructions, weights, biases, and input activations.
//     - Full program execution: LOAD -> DENSE -> DENSE -> STORE -> END.
//     - In-place standalone activation: LOAD -> ACT -> STORE -> END.
//     - Handshake timing and error reporting for illegal opcodes.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_top;

  localparam int W  = 4;
  localparam int AW = 32;

  logic clk;
  logic rst_n;

  // Execution Control
  logic        start;
  logic        busy;
  logic        done;
  logic        error;
  logic [7:0]  error_code;

  // Host Interface
  logic        host_wr_en;
  logic        host_rd_en;
  logic [2:0]  host_target_id;
  logic [15:0] host_addr;
  logic [31:0] host_wr_data32;
  logic signed [7:0] host_wr_data8;
  logic [31:0] host_rd_data32;
  logic signed [7:0] host_rd_data8;

  integer fail_count = 0;

  // Clock generation (100 MHz)
  always #5 clk = ~clk;

  // ---------------------------------------------------------------------------
  // Top-Level DUT Instantiation
  // ---------------------------------------------------------------------------
  tinyml_accelerator_top #(
    .IMEM_DEPTH_WORDS (1024),
    .SPAD_DEPTH       (4096),
    .PARAM_DEPTH      (65536),
    .OUT_DEPTH        (4096),
    .HMEM_DEPTH       (4096),
    .SIMD_WIDTH       (W),
    .ACC_WIDTH        (AW),
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

  // ---------------------------------------------------------------------------
  // Host Helper Tasks
  // ---------------------------------------------------------------------------

  task automatic write_imem(input [15:0] addr, input [31:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd0; // IMEM
    host_addr      = addr;
    host_wr_data32 = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic write_hmem(input [15:0] addr, input logic signed [7:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd1; // HMEM
    host_addr      = addr;
    host_wr_data8  = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic write_weight(input [15:0] addr, input logic signed [7:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd3; // WEIGHT
    host_addr      = addr;
    host_wr_data8  = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic write_bias(input [15:0] addr, input logic signed [7:0] data);
    @(negedge clk);
    host_wr_en     = 1'b1;
    host_target_id = 3'd4; // BIAS
    host_addr      = addr;
    host_wr_data8  = data;
    @(posedge clk);
    @(negedge clk);
    host_wr_en     = 1'b0;
  endtask

  task automatic read_outmem(input [15:0] addr, output logic signed [7:0] data);
    @(negedge clk);
    host_rd_en     = 1'b1;
    host_target_id = 3'd5; // OUTMEM
    host_addr      = addr;
    @(posedge clk);
    #1;
    data = host_rd_data8;
    @(negedge clk);
    host_rd_en     = 1'b0;
  endtask

  task automatic run_program(input string label, input int timeout_cycles);
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
      $error("[%s] TIMEOUT after %0d cycles", label, timeout_cycles);
      fail_count++;
    end else if (error) begin
      $error("[%s] Execution ERRORED with code=0x%02x", label, error_code);
      fail_count++;
    end else begin
      $display("[%s] Successfully finished in %0d clock cycles.", label, cycles);
    end
  endtask

  // ---------------------------------------------------------------------------
  // Test Sequence
  // ---------------------------------------------------------------------------
  initial begin
    logic signed [7:0] rd_val;
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

    // Reset pulse
    #20 rst_n = 1;
    #20;

    // =========================================================================
    // TEST 1: End-to-End 2-Layer Neural Network
    // =========================================================================
    // Layer 1: 4 inputs -> 3 outputs with ReLU
    // Layer 2: 3 inputs -> 2 outputs without activation
    //
    // Program:
    //   0: LOAD  mem_addr=0, sp_addr=0, len=4
    //   3: DENSE in=0, wt=0, out=64, bias=0, in_len=4, out_len=3, act=RELU, shift=-1, m0=0x40000000 (identity)
    //   8: DENSE in=64, wt=32, out=80, bias=16, in_len=3, out_len=2, act=NONE, shift=-1, m0=0x40000000
    //  13: STORE sp_addr=80, mem_addr=0, len=2
    //  16: END
    // =========================================================================
    $display("\n========================================================");
    $display("TEST 1: End-to-End 2-Layer Neural Network Inference");
    $display("========================================================");

    // 1. Preload Input x = [1, -2, 3, 2] into Host Data Memory at 0..3
    write_hmem(16'd0,  8'sd1);
    write_hmem(16'd1, -8'sd2);
    write_hmem(16'd2,  8'sd3);
    write_hmem(16'd3,  8'sd2);

    // 2. Preload Layer 1 Weights (4x3 = 12 bytes at weight_addr=0)
    // W1 rows: [1,2,-1,1], [-2,1,1,0], [3,-1,0,2]
    write_weight(16'd0,   8'sd1); write_weight(16'd1,   8'sd2); write_weight(16'd2,  -8'sd1); write_weight(16'd3,   8'sd1);
    write_weight(16'd4,  -8'sd2); write_weight(16'd5,   8'sd1); write_weight(16'd6,   8'sd1); write_weight(16'd7,   8'sd0);
    write_weight(16'd8,   8'sd3); write_weight(16'd9,  -8'sd1); write_weight(16'd10,  8'sd0); write_weight(16'd11,  8'sd2);

    // Layer 1 Biases (3 bytes at bias_addr=0): [5, 1, -2]
    write_bias(16'd0,  8'sd5);
    write_bias(16'd1,  8'sd1);
    write_bias(16'd2, -8'sd2);

    // 3. Preload Layer 2 Weights (3x2 = 6 bytes at weight_addr=32)
    // W2 rows: [2,-1,3], [-1,4,-2]
    write_weight(16'd32,  8'sd2); write_weight(16'd33, -8'sd1); write_weight(16'd34,  8'sd3);
    write_weight(16'd35, -8'sd1); write_weight(16'd36,  8'sd4); write_weight(16'd37, -8'sd2);

    // Layer 2 Biases (2 bytes at bias_addr=16): [-3, 5]
    write_bias(16'd16, -8'sd3);
    write_bias(16'd17,  8'sd5);

    // 4. Preload Instructions into Instruction Memory
    // Instr 0: LOAD (mem_addr=0, sp_addr=0, len=4)
    write_imem(16'd0, {OP_LOAD, 8'h00, ACT_NONE, 8'h00}); // Header
    write_imem(16'd1, {16'd0, 16'd0});                    // {mem_addr, sp_addr}
    write_imem(16'd2, {16'd4, 16'd0});                    // {length, reserved}

    // Instr 1: DENSE 1 (in=0, wt=0, out=64, bias=0, in_len=4, out_len=3, act=RELU, shift=-1)
    write_imem(16'd3, {OP_DENSE, 8'h00, ACT_RELU, 8'hFF}); // Header (shift = -1 = 8'hFF)
    write_imem(16'd4, {16'd0, 16'd0});                     // {in_addr, wt_addr}
    write_imem(16'd5, {16'd64, 16'd0});                    // {out_addr, bias_addr}
    write_imem(16'd6, {16'd4, 16'd3});                     // {in_len, out_len}
    write_imem(16'd7, 32'h4000_0000);                      // m0 = 0x40000000 (identity with shift=-1)

    // Instr 2: DENSE 2 (in=64, wt=32, out=80, bias=16, in_len=3, out_len=2, act=NONE, shift=-1)
    write_imem(16'd8,  {OP_DENSE, 8'h00, ACT_NONE, 8'hFF});// Header
    write_imem(16'd9,  {16'd64, 16'd32});                   // {in_addr, wt_addr}
    write_imem(16'd10, {16'd80, 16'd16});                   // {out_addr, bias_addr}
    write_imem(16'd11, {16'd3, 16'd2});                     // {in_len, out_len}
    write_imem(16'd12, 32'h4000_0000);                      // m0

    // Instr 3: STORE (sp_addr=80, mem_addr=0, len=2)
    write_imem(16'd13, {OP_STORE, 8'h00, ACT_NONE, 8'h00});// Header
    write_imem(16'd14, {16'd80, 16'd0});                    // {sp_addr, mem_addr}
    write_imem(16'd15, {16'd2, 16'd0});                     // {length, reserved}

    // Instr 4: END
    write_imem(16'd16, {OP_END, 8'h00, ACT_NONE, 8'h00});

    // Print IMEM
    $display("=== IMEM CONTENTS ===");
    for (int i = 0; i <= 16; i++) begin
      $display("imem[%0d] = 0x%08x", i, dut.u_imem.mem[i]);
    end

    // Execute program
    run_program("TEST1_TWO_LAYER_NN", 500);

    // Verify output memory
    read_outmem(16'd0, rd_val);
    $display("Output y[0]: got = %0d, expected = 20", rd_val);
    if (rd_val !== 8'sd20) begin
      $error("TEST 1: y[0] mismatch! Expected 20, got %0d", rd_val);
      fail_count++;
    end

    read_outmem(16'd1, rd_val);
    $display("Output y[1]: got = %0d, expected = -10", rd_val);
    if (rd_val !== -8'sd10) begin
      $error("TEST 1: y[1] mismatch! Expected -10, got %0d", rd_val);
      fail_count++;
    end

    // =========================================================================
    // TEST 2: Standalone ACT & Memory Copy
    // =========================================================================
    // Program:
    //   0: LOAD  mem_addr=10, sp_addr=100, len=4
    //   3: ACT   sp_addr=100, len=4, act=RELU
    //   5: STORE sp_addr=100, mem_addr=10, len=4
    //   8: END
    // =========================================================================
    $display("\n========================================================");
    $display("TEST 2: Standalone ACT (ReLU) and Pipelined Copy");
    $display("========================================================");

    // Reset accelerator
    rst_n = 0; #10; rst_n = 1; #10;

    // Preload host data at 10..13: [-15, 25, -2, 70]
    write_hmem(16'd10, -8'sd15);
    write_hmem(16'd11,  8'sd25);
    write_hmem(16'd12, -8'sd2);
    write_hmem(16'd13,  8'sd70);

    // Preload instructions
    write_imem(16'd0, {OP_LOAD, 8'h00, ACT_NONE, 8'h00});
    write_imem(16'd1, {16'd10, 16'd100});
    write_imem(16'd2, {16'd4, 16'd0});

    write_imem(16'd3, {OP_ACT, 8'h00, ACT_RELU, 8'h00});
    write_imem(16'd4, {16'd100, 16'd4});

    write_imem(16'd5, {OP_STORE, 8'h00, ACT_NONE, 8'h00});
    write_imem(16'd6, {16'd100, 16'd10});
    write_imem(16'd7, {16'd4, 16'd0});

    write_imem(16'd8, {OP_END, 8'h00, ACT_NONE, 8'h00});

    run_program("TEST2_STANDALONE_ACT", 300);

    // Verify output memory at 10..13: [0, 25, 0, 70]
    read_outmem(16'd10, rd_val);
    if (rd_val !== 8'sd0) begin $error("TEST 2: idx 0 got %0d, expected 0", rd_val); fail_count++; end
    read_outmem(16'd11, rd_val);
    if (rd_val !== 8'sd25) begin $error("TEST 2: idx 1 got %0d, expected 25", rd_val); fail_count++; end
    read_outmem(16'd12, rd_val);
    if (rd_val !== 8'sd0) begin $error("TEST 2: idx 2 got %0d, expected 0", rd_val); fail_count++; end
    read_outmem(16'd13, rd_val);
    if (rd_val !== 8'sd70) begin $error("TEST 2: idx 3 got %0d, expected 70", rd_val); fail_count++; end

    $display("TEST 2 output verified: [0, 25, 0, 70]");

    // =========================================================================
    // TEST 3: Illegal Opcode Error Handling
    // =========================================================================
    $display("\n========================================================");
    $display("TEST 3: Illegal Opcode Detection and Error State");
    $display("========================================================");

    rst_n = 0; #10; rst_n = 1; #10;

    // Load bad opcode 0xEE at address 0
    write_imem(16'd0, {8'hEE, 8'h00, ACT_NONE, 8'h00});

    @(negedge clk);
    start = 1'b1;
    @(posedge clk);
    @(negedge clk);
    start = 1'b0;

    // Wait for error assertion
    repeat (15) @(posedge clk);
    if (!error) begin
      $error("TEST 3: Expected error signal to be high on illegal opcode!");
      fail_count++;
    end else begin
      $display("TEST 3: Error correctly asserted with error_code = 0x%02x", error_code);
    end

    // =========================================================================
    // Summary
    // =========================================================================
    $display("\n========================================================");
    if (fail_count == 0) begin
      $display(">>> ALL TOP-LEVEL INTEGRATION TESTS PASSED SUCCESSFULLY! <<<");
    end else begin
      $display(">>> %0d FAILURE(S) DETECTED IN INTEGRATION TESTBENCH! <<<", fail_count);
    end
    $display("========================================================\n");

    $finish;
  end

endmodule
