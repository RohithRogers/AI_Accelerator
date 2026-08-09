// ============================================================================
// File: accumulator.sv
// Description: Running register holding the 32-bit accumulated total.
//              Simplified to support clear and run accumulation.
// ============================================================================

import tinyml_pkg::*;

module accumulator #(
    parameter int ACC_WIDTH = 32
) (
    input  logic                        clk,        // System Clock
    input  logic                        rst_n,      // Active-Low Reset
    
    // Control Ports
    input  logic                        clear,      // Set high to reset accumulator to 0
    input  logic                        accumulate, // Set high to add current chunk sum
    
    // Data Ports
    input  logic signed [ACC_WIDTH-1:0] sum_in,     // 32-bit sum from adder tree
    output logic signed [ACC_WIDTH-1:0] acc_out     // Current running accumulated sum
);

    logic signed [ACC_WIDTH-1:0] acc_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= '0;
        end else if (clear) begin
            acc_reg <= '0; // Clear accumulator for the next output neuron
        end else if (accumulate) begin
            acc_reg <= acc_reg + sum_in;
        end
    end

    assign acc_out = acc_reg;

endmodule
