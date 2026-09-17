// =============================================================
// Streaming Convolution Engine (line-buffer based)
// -------------------------------------------------------------
// This replaces an im2col address-generator with a genuine
// line-buffer / sliding-window architecture:
//
//   - Pixels stream in ONE AT A TIME, in raster order, into a
//     bank of K row shift-registers (row_buf).
//   - Once K full rows have been streamed in, the KxK window
//     needed for a MAC is just row_buf[ki][out_col+kj] --
//     small, fixed, register-indexed offsets. There is NO
//     separate block that computes an address into a big 2D
//     image memory, so there is no im2col-style address logic
//     left to get wrong.
//   - After computing one full output row, the buffer shifts
//     down by one row (row_buf[i] <= row_buf[i+1]), discarding
//     the oldest row and making room for the next incoming row.
//
// This mirrors how real streaming CNN accelerators avoid
// storing/re-addressing the whole image: only K rows are ever
// resident at once.
//
// NOTE: input_mem below stands in for a streaming source (e.g.
// a DMA/AXI-stream feed). In this module it's just an internal
// array the FSM reads sequentially -- swap S_FILL_PIXEL's read
// for your real input interface later; nothing else changes.
// =============================================================

module conv_engine_streaming #(
    parameter IMG_H       = 4,
    parameter IMG_W       = 4,
    parameter K           = 3,
    parameter DATA_WIDTH  = 8,
    parameter ACC_WIDTH   = 16,
    parameter OUT_H       = IMG_H - K + 1,
    parameter OUT_W       = IMG_W - K + 1
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    output reg  done
);

    // ---------------------------------------------------------
    // Memories
    // ---------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_mem  [0:IMG_H-1][0:IMG_W-1]; // simulated stream source
    reg signed [DATA_WIDTH-1:0] kernel_mem [0:K-1][0:K-1];
    reg signed [ACC_WIDTH-1:0]  output_mem [0:OUT_H-1][0:OUT_W-1];

    // The line buffer itself: K full rows, each IMG_W wide.
    // row_buf[K-1] = most recently completed row
    // row_buf[0]   = oldest row still needed for the window
    reg signed [DATA_WIDTH-1:0] row_buf [0:K-1][0:IMG_W-1];

    // ---------------------------------------------------------
    // FSM states
    // ---------------------------------------------------------
    localparam S_IDLE        = 3'd0;
    localparam S_FILL_PIXEL  = 3'd1; // stream one pixel into row_buf[K-1]
    localparam S_ROW_DONE    = 3'd2; // decide: compute a row, or just shift
    localparam S_COMPUTE_MAC = 3'd3; // accumulate one KxK window
    localparam S_WRITE_PIXEL = 3'd4; // write finished output pixel
    localparam S_SHIFT_ROWS  = 3'd5; // slide the line buffer down by 1 row
    localparam S_DONE        = 3'd6;

    reg [2:0] state;

    // Streaming-in counters
    reg [$clog2(IMG_H+1)-1:0] in_row;
    reg [$clog2(IMG_W+1)-1:0] in_col;
    reg [$clog2(K+1)-1:0]     buf_row_count; // how many valid rows are loaded (saturates at K)

    // Output / window-walk counters
    reg [$clog2(OUT_H+1)-1:0] out_row;
    reg [$clog2(OUT_W+1)-1:0] out_col;
    reg [$clog2(K+1)-1:0]     ki, kj;
    reg signed [ACC_WIDTH-1:0] acc;

    integer i; // for the row-shift loop

    // Current MAC tap: fixed small offsets into the line buffer,
    // NOT a computed address into a large image memory.
    wire signed [ACC_WIDTH-1:0] mac_product;
    assign mac_product = row_buf[ki][out_col + kj] * kernel_mem[ki][kj];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            in_row        <= 0;
            in_col        <= 0;
            buf_row_count <= 0;
            out_row       <= 0;
            out_col       <= 0;
            ki            <= 0;
            kj            <= 0;
            acc           <= 0;
        end else begin
            case (state)

                // -----------------------------------------
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        in_row        <= 0;
                        in_col        <= 0;
                        buf_row_count <= 0;
                        out_row       <= 0;
                        state         <= S_FILL_PIXEL;
                    end
                end

                // -----------------------------------------
                // Stream one pixel per cycle into the newest
                // row slot. No address computation beyond a
                // simple counter.
                S_FILL_PIXEL: begin
                    row_buf[K-1][in_col] <= input_mem[in_row][in_col];

                    if (in_col == IMG_W-1) begin
                        in_col <= 0;
                        state  <= S_ROW_DONE;
                    end else begin
                        in_col <= in_col + 1;
                    end
                end

                // -----------------------------------------
                // A row just finished streaming in. If we
                // already have K valid rows, compute one full
                // output row before shifting; otherwise just
                // shift (still accumulating rows).
                S_ROW_DONE: begin
                    if (buf_row_count == K) begin
                        out_col <= 0;
                        ki      <= 0;
                        kj      <= 0;
                        acc     <= 0;
                        state   <= S_COMPUTE_MAC;
                    end else begin
                        state <= S_SHIFT_ROWS;
                    end
                end

                // -----------------------------------------
                // One MAC per cycle over the KxK window at
                // the current output column.
                S_COMPUTE_MAC: begin
                    acc <= acc + mac_product;

                    if (kj == K-1) begin
                        kj <= 0;
                        if (ki == K-1) begin
                            ki    <= 0;
                            state <= S_WRITE_PIXEL;
                        end else begin
                            ki <= ki + 1;
                        end
                    end else begin
                        kj <= kj + 1;
                    end
                end

                // -----------------------------------------
                S_WRITE_PIXEL: begin
                    output_mem[out_row][out_col] <= acc;

                    if (out_col == OUT_W-1) begin
                        out_row <= out_row + 1;
                        state   <= S_SHIFT_ROWS;
                    end else begin
                        out_col <= out_col + 1;
                        ki      <= 0;
                        kj      <= 0;
                        acc     <= 0;
                        state   <= S_COMPUTE_MAC;
                    end
                end

                // -----------------------------------------
                // Slide the line buffer down by one row,
                // discarding the oldest row and freeing the
                // top slot for the next incoming row.
                S_SHIFT_ROWS: begin
                    for (i = 0; i < K-1; i = i + 1)
                        row_buf[i] <= row_buf[i+1];

                    if (buf_row_count < K)
                        buf_row_count <= buf_row_count + 1;

                    in_col <= 0;

                    if (in_row == IMG_H-1) begin
                        state <= S_DONE;
                    end else begin
                        in_row <= in_row + 1;
                        state  <= S_FILL_PIXEL;
                    end
                end

                // -----------------------------------------
                S_DONE: begin
                    done <= 1'b1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule


// =============================================================
// Self-checking testbench
// -------------------------------------------------------------
// Same 4x4 image / 3x3 all-ones kernel as the earlier version,
// so you can diff results directly:
//
//   1  2  3  4
//   5  6  7  8
//   9 10 11 12
//  13 14 15 16
//
// Expected 2x2 output:
//   54  63
//   90  99
// =============================================================
module tb_conv_engine_streaming;

    reg clk;
    reg rst;
    reg start;
    wire done;

    integer i, j;
    integer errors;

    conv_engine_streaming #(
        .IMG_H(4), .IMG_W(4), .K(3),
        .DATA_WIDTH(8), .ACC_WIDTH(16)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done)
    );

    always #5 clk = ~clk;

    reg signed [15:0] expected [0:1][0:1];

    initial begin
        clk    = 0;
        rst    = 1;
        start  = 0;
        errors = 0;

        // Load the "streaming source" image
        dut.input_mem[0][0] = 1;  dut.input_mem[0][1] = 2;
        dut.input_mem[0][2] = 3;  dut.input_mem[0][3] = 4;
        dut.input_mem[1][0] = 5;  dut.input_mem[1][1] = 6;
        dut.input_mem[1][2] = 7;  dut.input_mem[1][3] = 8;
        dut.input_mem[2][0] = 9;  dut.input_mem[2][1] = 10;
        dut.input_mem[2][2] = 11; dut.input_mem[2][3] = 12;
        dut.input_mem[3][0] = 13; dut.input_mem[3][1] = 14;
        dut.input_mem[3][2] = 15; dut.input_mem[3][3] = 16;

        // Kernel: all ones
        for (i = 0; i < 3; i = i + 1)
            for (j = 0; j < 3; j = j + 1)
                dut.kernel_mem[i][j] = 1;

        expected[0][0] = 54; expected[0][1] = 63;
        expected[1][0] = 90; expected[1][1] = 99;

        #12 rst = 0;
        #10 start = 1;
        #10 start = 0;

        wait (done == 1);
        #5;

        for (i = 0; i < 2; i = i + 1) begin
            for (j = 0; j < 2; j = j + 1) begin
                if (dut.output_mem[i][j] !== expected[i][j]) begin
                    $display("MISMATCH at (%0d,%0d): got %0d, expected %0d",
                              i, j, dut.output_mem[i][j], expected[i][j]);
                    errors = errors + 1;
                end else begin
                    $display("OK       (%0d,%0d): %0d", i, j, dut.output_mem[i][j]);
                end
            end
        end

        if (errors == 0)
            $display("ALL TESTS PASSED");
        else
            $display("%0d TEST(S) FAILED", errors);

        $finish;
    end

endmodule
