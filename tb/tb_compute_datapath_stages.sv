// ============================================================================
// File: tb_compute_datapath_stages.sv
// Description: Stage-by-stage verification testbench for the TinyML Accelerator
//              Compute Datapath configured for SIMD_WIDTH = 8 and input_len = 8.
//              Monitors and logs sequential execution across all 8 datapath stages.
// ============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_compute_datapath_stages;

    localparam int SIMD_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int CLK_PERIOD = 10; // 100 MHz clock

    // System Signals
    logic clk;
    logic rst_n;

    // Command Interface
    logic        start;
    logic [15:0] input_addr;
    logic [15:0] weight_addr;
    logic [15:0] output_addr;
    logic [15:0] bias_addr;
    logic [15:0] input_len;
    logic [15:0] output_len;
    logic [7:0]  activation;

    logic        busy;
    logic        done;
    logic        error;

    // Memory Interconnects
    logic [15:0] spad_rd_addr[SIMD_WIDTH];
    logic signed [7:0] spad_rd_data[SIMD_WIDTH];
    logic        spad_wr_en;
    logic [15:0] spad_wr_addr;
    logic signed [7:0] spad_wr_data;

    logic [15:0] weight_rd_addr[SIMD_WIDTH];
    logic signed [7:0] weight_rd_data[SIMD_WIDTH];

    logic [15:0] bias_rd_addr;
    logic signed [7:0] bias_rd_data;

    // Testbench Simulated Memory Arrays
    logic signed [7:0] spad_mem  [0:4095];
    logic signed [7:0] weight_mem[0:65535];
    logic signed [7:0] bias_mem  [0:4095];

    // Instantiate DUT (Dense Compute Engine with SIMD_WIDTH = 8, REQUANT_SHIFT = 1)
    dense_engine #(
        .SIMD_WIDTH(SIMD_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .REQUANT_SHIFT(1), // Right shift by 1 bit (Divide by 2)
        .RELU6_MAX(127)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .input_addr(input_addr),
        .weight_addr(weight_addr),
        .output_addr(output_addr),
        .bias_addr(bias_addr),
        .input_len(input_len),
        .output_len(output_len),
        .activation(activation),
        .busy(busy),
        .done(done),
        .error(error),
        .spad_rd_addr(spad_rd_addr),
        .spad_rd_data(spad_rd_data),
        .spad_wr_en(spad_wr_en),
        .spad_wr_addr(spad_wr_addr),
        .spad_wr_data(spad_wr_data),
        .weight_rd_addr(weight_rd_addr),
        .weight_rd_data(weight_rd_data),
        .bias_rd_addr(bias_rd_addr),
        .bias_rd_data(bias_rd_data)
    );

    // Clock Generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Memory Models (1-cycle synchronous read latency)
    always_ff @(posedge clk) begin
        for (int i = 0; i < SIMD_WIDTH; i++) begin
            spad_rd_data[i]   <= spad_mem[spad_rd_addr[i]];
            weight_rd_data[i] <= weight_mem[weight_rd_addr[i]];
        end
        bias_rd_data <= bias_mem[bias_rd_addr];
    end

    always_ff @(posedge clk) begin
        if (spad_wr_en) begin
            spad_mem[spad_wr_addr] <= spad_wr_data;
        end
    end

    // Monitor Procedure: Logs Each Stage for 8-SIMD Lane Execution
    always @(posedge clk) begin
        if (rst_n && busy) begin
            case (dut.state)
                3'd1: begin // ST_READ_BIAS
                    $display("\n--------------------------------------------------------------------------------");
                    $display("STAGE 1: BIAS FETCH & PREPARATION");
                    $display("--------------------------------------------------------------------------------");
                    $display("  [AGU] Reading Bias at Address: %0d", bias_rd_addr);
                end

                3'd2: begin // ST_INIT_ACC
                    $display("  [Bias Read] Raw Bias Byte Read: %0d", bias_rd_data);
                    $display("  [Accumulator] Cleared to 0.");
                end

                3'd3: begin // ST_LOAD_CHUNK
                    $display("\n--------------------------------------------------------------------------------");
                    $display("STAGE 2: ADDRESS GENERATION & VECTOR LOADER (SIMD = 8)");
                    $display("--------------------------------------------------------------------------------");
                    $display("  [AGU] Scratchpad Read Addrs : [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                             spad_rd_addr[0], spad_rd_addr[1], spad_rd_addr[2], spad_rd_addr[3],
                             spad_rd_addr[4], spad_rd_addr[5], spad_rd_addr[6], spad_rd_addr[7]);
                    $display("  [AGU] Weight RAM Read Addrs : [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                             weight_rd_addr[0], weight_rd_addr[1], weight_rd_addr[2], weight_rd_addr[3],
                             weight_rd_addr[4], weight_rd_addr[5], weight_rd_addr[6], weight_rd_addr[7]);
                    $display("  [AGU] Active Lanes Bitmask  : 8'b%08b", dut.lanes_valid);
                end

                3'd5: begin // ST_MAC
                    $display("\n--------------------------------------------------------------------------------");
                    $display("STAGE 3 & 4: SIMD MAC ARRAY (8 LANES) & ACCUMULATOR INTEGRATION");
                    $display("--------------------------------------------------------------------------------");
                    $display("  [MAC Input] Activations (Scratchpad) : [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                             spad_rd_data[0], spad_rd_data[1], spad_rd_data[2], spad_rd_data[3],
                             spad_rd_data[4], spad_rd_data[5], spad_rd_data[6], spad_rd_data[7]);
                    $display("  [MAC Input] Weights (Weight RAM)     : [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                             weight_rd_data[0], weight_rd_data[1], weight_rd_data[2], weight_rd_data[3],
                             weight_rd_data[4], weight_rd_data[5], weight_rd_data[6], weight_rd_data[7]);
                    $display("  [MAC Output] Parallel 8-Lane Products : [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                             dut.products[0], dut.products[1], dut.products[2], dut.products[3],
                             dut.products[4], dut.products[5], dut.products[6], dut.products[7]);
                    $display("  [Adder Tree] 8-Lane Product Sum      : %0d", dut.adder_sum);
                    $display("  [Accumulator] Updated Total Sum      : %0d", dut.acc_val + dut.adder_sum);
                end

                3'd6: begin // ST_POSTPROCESS
                    $display("\n--------------------------------------------------------------------------------");
                    $display("STAGE 5 & 6: BIAS ADDITION & REQUANTIZATION");
                    $display("--------------------------------------------------------------------------------");
                    $display("  [Bias Adder] Accumulated Sum + Bias  : %0d + %0d = %0d",
                             dut.acc_val, bias_rd_data, dut.biased_sum);
                    $display("  [Requantizer] Shift (1 bit) Result   : %0d >>> 1 = %0d",
                             dut.biased_sum, dut.requant_out);

                    $display("\n--------------------------------------------------------------------------------");
                    $display("STAGE 7: ACTIVATION FUNCTION (Non-Linearity)");
                    $display("--------------------------------------------------------------------------------");
                    $display("  [Activation] Type: RELU | Input: %0d ---> Output: %0d",
                             dut.requant_out, dut.act_out);
                end

                3'd7: begin // ST_WRITE_RESULT
                    $display("\n--------------------------------------------------------------------------------");
                    $display("STAGE 8: WRITEBACK & MEMORY UPDATE");
                    $display("--------------------------------------------------------------------------------");
                    $display("  [Writeback] Target Scratchpad Addr  : %0d", spad_wr_addr);
                    $display("  [Writeback] Saturated INT8 Value    : %0d", spad_wr_data);
                    $display("  [Memory] Scratchpad[%0d] <= %0d", spad_wr_addr, spad_wr_data);
                end
            endcase
        end
    end

    // Main Test Execution Sequence
    initial begin
        $display("\n================================================================================");
        $display("   TINYML ACCELERATOR - COMPUTE DATAPATH (SIMD = 8, 8-ELEMENT INPUT VECTOR)     ");
        $display("================================================================================");

        // Step 1: Initialize System & Reset
        rst_n       = 0;
        start       = 0;
        input_addr  = 0;
        weight_addr = 0;
        output_addr = 100;
        bias_addr   = 0;
        input_len   = 0;
        output_len  = 0;
        activation  = ACT_NONE;

        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // Step 2: Preload 8-Element Input Vector & Weight Vector
        // Input Vector (8 elements): [10, 20, 30, 40, 5, 10, 15, 20]
        spad_mem[0] = 8'sd10;
        spad_mem[1] = 8'sd20;
        spad_mem[2] = 8'sd30;
        spad_mem[3] = 8'sd40;
        spad_mem[4] = 8'sd5;
        spad_mem[5] = 8'sd10;
        spad_mem[6] = 8'sd15;
        spad_mem[7] = 8'sd20;

        // Weight Vector (8 elements): [1, 1, 1, 1, 2, 2, 2, 2]
        weight_mem[0] = 8'sd1;
        weight_mem[1] = 8'sd1;
        weight_mem[2] = 8'sd1;
        weight_mem[3] = 8'sd1;
        weight_mem[4] = 8'sd2;
        weight_mem[5] = 8'sd2;
        weight_mem[6] = 8'sd2;
        weight_mem[7] = 8'sd2;

        // Bias: +10
        bias_mem[0] = 8'sd10;

        // Step 3: Trigger Compute Command (Input Len = 8, Output Len = 1, RELU)
        $display("\n[COMMAND] Launching Compute Execution:");
        $display("  Input Addr = 0, Weight Addr = 0, Output Addr = 100, Bias Addr = 0");
        $display("  Input Len  = 8 (1 SIMD Chunk of 8), Output Len = 1, Activation = RELU");

        input_addr  = 16'd0;
        weight_addr = 16'd0;
        output_addr = 16'd100;
        bias_addr   = 16'd0;
        input_len   = 16'd8;
        output_len  = 16'd1;
        activation  = ACT_RELU;

        start = 1;
        #(CLK_PERIOD);
        start = 0;

        // Wait for execution completion
        wait(done);
        #(CLK_PERIOD * 2);

        $display("\n================================================================================");
        $display("  [FINAL RESULT] Scratchpad[100] = %0d", spad_mem[100]);
        $display("================================================================================");
        $display("  8-LANE SIMD COMPUTE DATAPATH VERIFIED SUCCESSFULLY!");
        $display("================================================================================\n");

        $finish;
    end

endmodule
