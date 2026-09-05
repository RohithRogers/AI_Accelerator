// ============================================================================
// File: vector_loader.sv
// Description: Address generator unit (AGU) for loading inputs and weights.
//              Generates addresses and handles masking for tail elements.
//              Controlled by step strobe next_chunk from dense_engine.
// ============================================================================

import tinyml_pkg::*;

module vector_loader #(
    parameter int SIMD_WIDTH = 4 // Processing lanes (W=4)
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Command Handshake
    input  logic        start,            // Set high to launch next loop sequence
    input  logic        next_chunk,       // Pulse to advance to next chunk
    input  logic [15:0] base_sp_addr,     // Input activation start address in Scratchpad
    input  logic [15:0] base_weight_addr, // Weights start address for current output neuron
    input  logic [15:0] input_len,        // Vector length (N)
    
    output logic        busy,             // Set high while actively fetching chunks
    output logic        chunk_valid,      // High when output addresses are stable
    
    // Generated Address Buses
    output logic [15:0] spad_rd_addr    [0:SIMD_WIDTH-1], // Scratchpad read addresses
    output logic [15:0] weight_rd_addr[0:SIMD_WIDTH-1], // Weight RAM read addresses
    
    // Parallel Lane Mask Outputs
    output logic [SIMD_WIDTH-1:0] lanes_valid, // Tells MAC which lanes contain valid numbers
    output logic        last_chunk            // Set high on the final chunk of the loop
);

    logic [15:0] chunk_offset; // Running counter of inputs processed

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            chunk_offset <= 16'd0;
            busy         <= 1'b0;
            chunk_valid  <= 1'b0;
        end else if (start && !busy) begin
            // Initialize pointers and start execution
            chunk_offset <= 16'd0;
            busy         <= 1'b1;
            chunk_valid  <= 1'b1;
        end else if (busy) begin
            if (next_chunk) begin
                // Increment chunk offset by SIMD_WIDTH when requested
                if (chunk_offset + SIMD_WIDTH < input_len) begin
                    chunk_offset <= chunk_offset + SIMD_WIDTH;
                    chunk_valid  <= 1'b1;
                end else begin
                    // Reached the end of the loop, return to idle
                    busy        <= 1'b0;
                    chunk_valid <= 1'b0;
                end
            end
        end else begin
            chunk_valid <= 1'b0;
        end
    end

    // Continuous per-lane drivers
    generate
        for (genvar lane = 0; lane < SIMD_WIDTH; lane++) begin : g_lane_addr
            assign spad_rd_addr[lane]   = base_sp_addr + chunk_offset + lane;
            assign weight_rd_addr[lane] = base_weight_addr + chunk_offset + lane;
            assign lanes_valid[lane]    = busy && (chunk_offset + lane < input_len);
        end
    endgenerate
    assign last_chunk = (chunk_offset + SIMD_WIDTH >= input_len);

endmodule
