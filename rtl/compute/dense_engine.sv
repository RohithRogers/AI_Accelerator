// ============================================================================
// File: dense_engine.sv
// Description: Core Compute Engine FSM. Controls the datapath execution loop,
//              instantiates sub-blocks, and manages loop counters.
// ============================================================================

import tinyml_pkg::*;

module dense_engine #(
    parameter int SIMD_WIDTH     = 4,
    parameter int ACC_WIDTH      = 32,
    parameter int REQUANT_SHIFT  = 0,
    parameter int RELU6_MAX      = 127
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // FSM Control Command Ports
    input  logic        start,
    input  logic [15:0] input_addr,
    input  logic [15:0] weight_addr,
    input  logic [15:0] output_addr,
    input  logic [15:0] bias_addr,
    input  logic [15:0] input_len,
    input  logic [15:0] output_len,
    input  logic [7:0]  activation,
    input  logic [31:0] requant_m0,
    input  logic signed [7:0] requant_shift,
    
    output logic        busy,
    output logic        done,
    output logic        error,
    
    // Scratchpad Memory Read Interface
    output logic [SIMD_WIDTH*16-1:0] spad_rd_addr_bus,
    input  logic signed [SIMD_WIDTH*8-1:0] spad_rd_data_bus,
    
    // Scratchpad Memory Write Interface
    output logic        spad_wr_en,
    output logic [15:0] spad_wr_addr,
    output logic signed [7:0] spad_wr_data,
    
    // Weight RAM Interface
    output logic [SIMD_WIDTH*16-1:0] weight_rd_addr_bus,
    input  logic signed [SIMD_WIDTH*8-1:0] weight_rd_data_bus,
    
    // Bias RAM Interface
    output logic [15:0] bias_rd_addr,
    input  logic signed [7:0] bias_rd_data
);

    // Compute Engine Sub-FSM States
    typedef enum logic [3:0] {
        ST_IDLE,
        ST_READ_BIAS,
        ST_INIT_ACC,
        ST_LOAD_CHUNK,
        ST_WAIT_MEM,
        ST_MAC,
        ST_POSTPROCESS,
        ST_WRITE_RESULT,
        ST_DONE
    } state_t;

    state_t state, next_state;

    // Loop Counters
    logic [15:0] out_idx;
    logic [15:0] cur_weight_base;
    logic [15:0] chunk_offset;

    // Datapath Control Interconnects
    logic [SIMD_WIDTH-1:0] lanes_valid;
    logic        last_chunk;

    logic signed [ACC_WIDTH-1:0] adder_sum;
    
    logic        acc_clear;
    logic        acc_accumulate;
    logic signed [ACC_WIDTH-1:0] acc_val;

    logic signed [ACC_WIDTH-1:0] biased_sum;
    logic signed [7:0] requant_out;
    logic signed [7:0] act_out;

    logic        wb_enable;
    logic [15:0] wb_addr;
    logic signed [7:0] wb_data;

    // Address generation is kept local to avoid unresolved unpacked-array
    // outputs across a nested hierarchy in Icarus Verilog.
    generate
        for (genvar lane = 0; lane < SIMD_WIDTH; lane++) begin : g_lane_addr
            assign spad_rd_addr_bus[lane*16 +: 16]   = input_addr + chunk_offset + lane;
            assign weight_rd_addr_bus[lane*16 +: 16] = cur_weight_base + chunk_offset + lane;
            assign lanes_valid[lane]    = chunk_offset + lane < input_len;
        end
    endgenerate
    assign last_chunk = chunk_offset + SIMD_WIDTH >= input_len;

    // Packed-bus SIMD MAC reduction.
    always_comb begin
        adder_sum = '0;
        for (int lane = 0; lane < SIMD_WIDTH; lane++)
            if (lanes_valid[lane])
                adder_sum += $signed(spad_rd_data_bus[lane*8 +: 8]) *
                             $signed(weight_rd_data_bus[lane*8 +: 8]);
    end

    // 4. Instantiate Accumulator
    accumulator #(
        .ACC_WIDTH(ACC_WIDTH)
    ) u_accumulator (
        .clk(clk),
        .rst_n(rst_n),
        .clear(acc_clear),
        .accumulate(acc_accumulate),
        .sum_in(adder_sum),
        .acc_out(acc_val)
    );

    // 5. Instantiate Bias Adder
    bias_adder #(
        .ACC_WIDTH(ACC_WIDTH)
    ) u_bias_adder (
        .acc_in(acc_val),
        .bias_in(bias_rd_data),
        .sum_out(biased_sum)
    );

    // 6. Instantiate Requantizer
    requantizer_runtime #(
        .ACC_WIDTH(ACC_WIDTH)
    ) u_requantizer (
        .acc_in(biased_sum),
        .m0(requant_m0),
        .shift_n(requant_shift),
        .val_out(requant_out)
    );

    // 7. Instantiate Activation Unit
    activation_unit #(
        .RELU6_MAX(RELU6_MAX)
    ) u_activation (
        .act_type(activation),
        .data_in(requant_out),
        .data_out(act_out)
    );

    // 8. Instantiate Writeback Unit
    writeback_unit u_writeback (
        .clk(clk),
        .rst_n(rst_n),
        .write_enable(wb_enable),
        .write_addr(wb_addr),
        .write_data(wb_data),
        .spad_wr_en(spad_wr_en),
        .spad_wr_addr(spad_wr_addr),
        .spad_wr_data(spad_wr_data)
    );

    // Loop Control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            out_idx <= 16'd0;
            cur_weight_base <= 16'd0;
            chunk_offset <= 16'd0;
        end else begin
            state <= next_state;
            if (state == ST_IDLE && start) begin
                out_idx <= 16'd0;
                cur_weight_base <= weight_addr;
            end else if (state == ST_INIT_ACC) begin
                chunk_offset <= 16'd0;
            end else if (state == ST_MAC && !last_chunk) begin
                chunk_offset <= chunk_offset + SIMD_WIDTH;
            end else if (state == ST_WRITE_RESULT) begin
                out_idx <= out_idx + 1'b1;
                cur_weight_base <= cur_weight_base + input_len;
            end
        end
    end

    // FSM Outputs
    always_comb begin
        next_state         = state;
        busy               = 1'b1;
        done               = 1'b0;
        error              = 1'b0;
        acc_clear          = 1'b0;
        acc_accumulate     = 1'b0;
        wb_enable          = 1'b0;
        wb_addr            = output_addr + out_idx;
        wb_data            = act_out;
        
        bias_rd_addr       = bias_addr + out_idx;

        case (state)
            ST_IDLE: begin
                busy = 1'b0;
                if (start) begin
                    if (input_len == 16'd0 || output_len == 16'd0) begin
                        error      = 1'b1;
                        next_state = ST_IDLE;
                    end else begin
                        next_state = ST_READ_BIAS;
                    end
                end
            end

            ST_READ_BIAS: begin
                next_state = ST_INIT_ACC;
            end

            ST_INIT_ACC: begin
                acc_clear     = 1'b1;
                next_state    = ST_LOAD_CHUNK;
            end

            ST_LOAD_CHUNK: begin
                next_state = ST_WAIT_MEM;
            end

            ST_WAIT_MEM: begin
                next_state = ST_MAC;
            end

            ST_MAC: begin
                acc_accumulate = 1'b1;
                if (last_chunk) begin
                    next_state = ST_POSTPROCESS;
                end else begin
                    next_state = ST_LOAD_CHUNK;
                end
            end

            ST_POSTPROCESS: begin
                next_state = ST_WRITE_RESULT;
            end

            ST_WRITE_RESULT: begin
                wb_enable = 1'b1;
                if (out_idx + 1'b1 >= output_len) begin
                    next_state = ST_DONE;
                end else begin
                    next_state = ST_READ_BIAS;
                end
            end

            ST_DONE: begin
                done       = 1'b1;
                busy       = 1'b0;
                next_state = ST_IDLE;
            end

            default: next_state = ST_IDLE;
        endcase
    end

endmodule
