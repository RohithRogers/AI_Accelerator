// ============================================================================
// File: requantizer.sv
// Description: Squeezes the internal 32-bit sum back down to signed INT8.
//              Applies an arithmetic right shift and saturates output to [-128, 127].
// ============================================================================

import tinyml_pkg::*;

module requantizer #(
    parameter int ACC_WIDTH     = 32,
    parameter int REQUANT_SHIFT = 0
) (
    input  logic signed [ACC_WIDTH-1:0] acc_in,
    output logic signed [7:0]           val_out
);

    logic signed [ACC_WIDTH-1:0] shifted_val;

    always_comb begin
        shifted_val = acc_in >>> REQUANT_SHIFT;
        val_out     = sat8(shifted_val); // Call standard saturation helper from package
    end

endmodule
