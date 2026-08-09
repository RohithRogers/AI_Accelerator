// ============================================================================
// File: bias_adder.sv
// Description: Adds the sign-extended 8-bit bias value to the accumulated sum.
// ============================================================================

import tinyml_pkg::*;

module bias_adder #(
    parameter int ACC_WIDTH = 32
) (
    input  logic signed [ACC_WIDTH-1:0] acc_in,  // Sum from accumulator
    input  logic signed [7:0]           bias_in, // 8-bit bias value
    output logic signed [ACC_WIDTH-1:0] sum_out  // Biased sum output
);

    always_comb begin
        // Sign-extend 8-bit bias to 32 bits and add to the accumulated sum
        sum_out = acc_in + $signed({{24{bias_in[7]}}, bias_in});
    end

endmodule
