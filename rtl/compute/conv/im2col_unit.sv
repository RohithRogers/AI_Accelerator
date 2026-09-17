// =============================================================================
// im2col_unit.sv — Hardware im2col Pre-Processor for Convolution
//
// PURPOSE:
//   Rearranges input feature maps stored in the scratchpad into a column matrix,
//   enabling convolution to be executed as a dense matrix-vector multiply by
//   the dense_engine.
//
// OPERATION (triggered by a CONV instruction in controller_fsm):
//   For each output pixel location (oy, ox) and each kernel element (ky, kx, c):
//     1. Compute the source pixel address: input_base + c*H*W + iy*W + ix
//     2. If the pixel is outside the input bounds (padding zone), write zero
//     3. Otherwise, read from scratchpad and write to column buffer at output_base
//   Output column buffer layout: [out_y * out_w + out_x][c * K * K + ky * K + kx]
//   Total column buffer size: out_h * out_w * C * K * K bytes
//
// INTERFACE:
//   input_base   — scratchpad base address of input feature map
//   output_base  — scratchpad base address to write the column matrix
//   img_width/height — spatial dimensions of input feature map
//   channels     — number of input channels (C)
//   kernel_size  — square kernel side length (K)
//   stride/padding — standard convolution parameters
//
// PORT NAMING matches tinyml_accelerator_top.sv instantiation:
//   .input_base, .output_base (not input_base_addr / output_base_addr)
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module im2col_unit #(
    parameter int unsigned SPAD_DEPTH = 4096
) (
    input  logic        clk,
    input  logic        rst_n,

    // Command handshake (from controller_fsm)
    input  logic        start,
    output logic        busy,
    output logic        done,

    // Geometry parameters (from conv_config_reg)
    input  logic [7:0]  img_width,       // W_in
    input  logic [7:0]  img_height,      // H_in
    input  logic [7:0]  channels,        // C_in
    input  logic [7:0]  kernel_h,
    input  logic [7:0]  kernel_w,
    input  logic [7:0]  stride,          // S
    input  logic [7:0]  padding,         // P

    // Base addresses (from controller_fsm CONV instruction operands)
    input  logic [15:0] input_base,      // SPAD base of input feature map
    input  logic [15:0] output_base,     // SPAD base of output column buffer

    // Scratchpad read interface
    output logic        spad_rd_en,
    output logic [15:0] spad_rd_addr,
    input  logic signed [7:0] spad_rd_data,

    // Scratchpad write interface
    output logic        spad_wr_en,
    output logic [15:0] spad_wr_addr,
    output logic signed [7:0] spad_wr_data
);

    // -------------------------------------------------------------------------
    // Output map dimensions (combinational)
    // -------------------------------------------------------------------------
    logic [15:0] out_w;
    logic [15:0] out_h;
    assign out_w = (stride == 0 || img_width + 2 * padding < kernel_w) ? 0 :
                   (img_width  + 2 * padding - kernel_w) / stride + 1;
    assign out_h = (stride == 0 || img_height + 2 * padding < kernel_h) ? 0 :
                   (img_height + 2 * padding - kernel_h) / stride + 1;

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_CALC,
        ST_READ,
        ST_WAIT_MEM,
        ST_WRITE,
        ST_DONE
    } state_t;

    state_t state;

    // Loop counters — iterate: ky -> kx -> c -> ox -> oy (inner to outer)
    logic [7:0] ox, oy;   // output spatial position
    logic [7:0] c;         // input channel
    logic [7:0] kx, ky;   // kernel position

    // Computed per-iteration
    logic signed [15:0] ix_s, iy_s;  // signed source pixel coords
    logic               is_pad;       // true when pixel is in padding zone
    logic [15:0]        src_addr;     // flat source address in SPAD
    logic [15:0]        dst_ptr;      // flat destination address in col buffer
    logic signed [7:0]  latched_val;  // pixel value after mem read

    // Sequential iteration: compute addresses combinationally, act in FSM
    assign ix_s = $signed({1'b0, ox}) * $signed({1'b0, stride}) +
                  $signed({1'b0, kx}) - $signed({1'b0, padding});
    assign iy_s = $signed({1'b0, oy}) * $signed({1'b0, stride}) +
                  $signed({1'b0, ky}) - $signed({1'b0, padding});

    assign is_pad  = (ix_s < 16'sd0) | (ix_s >= $signed({1'b0, img_width}))  |
                     (iy_s < 16'sd0) | (iy_s >= $signed({1'b0, img_height}));

    assign src_addr = input_base
                    + c * img_height * img_width
                    + $unsigned(iy_s) * img_width
                    + $unsigned(ix_s);

    // -------------------------------------------------------------------------
    // Sequential block
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            ox          <= 8'd0;
            oy          <= 8'd0;
            c           <= 8'd0;
            kx          <= 8'd0;
            ky          <= 8'd0;
            dst_ptr     <= 16'd0;
            latched_val <= 8'sd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        ox      <= 8'd0;
                        oy      <= 8'd0;
                        c       <= 8'd0;
                        kx      <= 8'd0;
                        ky      <= 8'd0;
                        dst_ptr <= output_base;
                        state   <= ST_CALC;
                    end
                end

                ST_CALC: begin
                    if (is_pad) begin
                        latched_val <= 8'sd0;
                        state       <= ST_WRITE;
                    end else begin
                        state <= ST_READ;
                    end
                end

                ST_READ: begin
                    // Issue read; will latch on next cycle (ST_WAIT_MEM)
                    state <= ST_WAIT_MEM;
                end

                ST_WAIT_MEM: begin
                    latched_val <= spad_rd_data;
                    state       <= ST_WRITE;
                end

                ST_WRITE: begin
                    dst_ptr <= dst_ptr + 16'd1;

                    // Patch layout is [channel][kernel_y][kernel_x].
                    if (kx + 8'd1 < kernel_w) begin
                        kx <= kx + 8'd1;
                        state <= ST_CALC;
                    end else begin
                        kx <= 8'd0;
                        if (ky + 8'd1 < kernel_h) begin
                            ky <= ky + 8'd1;
                            state <= ST_CALC;
                        end else begin
                            ky <= 8'd0;
                            if (c + 8'd1 < channels) begin
                                c <= c + 8'd1;
                                state <= ST_CALC;
                            end else begin
                                c <= 8'd0;
                                if (ox + 8'd1 < out_w) begin
                                    ox <= ox + 8'd1;
                                    state <= ST_CALC;
                                end else begin
                                    ox <= 8'd0;
                                    if (oy + 8'd1 < out_h) begin
                                        oy <= oy + 8'd1;
                                        state <= ST_CALC;
                                    end else begin
                                        state <= ST_DONE;
                                    end
                                end
                            end
                        end
                    end
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Combinational output driver
    // -------------------------------------------------------------------------
    always_comb begin
        busy         = (state != ST_IDLE);
        done         = (state == ST_DONE);
        spad_rd_en   = (state == ST_READ);
        spad_rd_addr = src_addr;
        spad_wr_en   = (state == ST_WRITE);
        spad_wr_addr = dst_ptr;
        spad_wr_data = latched_val;
    end

endmodule
