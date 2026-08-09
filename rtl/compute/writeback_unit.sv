// ============================================================================
// File: writeback_unit.sv
// Description: Manages writing calculated neuron outputs back to the Scratchpad.
// ============================================================================

import tinyml_pkg::*;

module writeback_unit (
    input  logic              clk,
    input  logic              rst_n,
    
    // Core Interface
    input  logic              write_enable, // Strobe signaling writeback is active
    input  logic [15:0]       write_addr,   // Target scratchpad destination address
    input  logic signed [7:0] write_data,   // Activated output activation byte
    
    // Scratchpad RAM Write Interface
    output logic              spad_wr_en,
    output logic [15:0]       spad_wr_addr,
    output logic signed [7:0] spad_wr_data
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spad_wr_en   <= 1'b0;
            spad_wr_addr <= 16'd0;
            spad_wr_data <= 8'd0;
        end else begin
            // Pass signals directly with a 1-cycle write registration
            spad_wr_en   <= write_enable;
            spad_wr_addr <= write_addr;
            spad_wr_data <= write_data;
        end
    end

endmodule
