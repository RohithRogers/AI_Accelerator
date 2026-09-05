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

    // Sign-extend 8-bit bias to 32 bits and add to the accumulated sum
    assign sum_out = acc_in + {{(ACC_WIDTH-8){bias_in[7]}}, bias_in};

endmodule
