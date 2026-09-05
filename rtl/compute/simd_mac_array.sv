// ============================================================================
// File: simd_mac_array.sv
// Description: Parallel multiplier array for the TinyML Accelerator.
//              Performs 4 parallel signed 8-bit multiplications in 1 cycle.
// ============================================================================

import tinyml_pkg::*;

module simd_mac_array #(
    parameter int SIMD_WIDTH = 4 // Number of parallel operations (default W=4)
) (
    // Inputs: 4 signed 8-bit inputs and 4 weights
    input  logic signed [7:0]     act_data   [0:SIMD_WIDTH-1], 
    input  logic signed [7:0]     weight_data[0:SIMD_WIDTH-1], 
    
    // Lane mask to selectively disable lanes for non-multiple-of-4 vector lengths
    input  logic [SIMD_WIDTH-1:0] lanes_valid, 
    
    // Outputs: 4 signed 16-bit multiplication products
    output logic signed [15:0]    products   [0:SIMD_WIDTH-1]  
);

    // Combinational block executing multiplications using generate for synthesis
    generate
        for (genvar i = 0; i < SIMD_WIDTH; i++) begin : gen_mac
            assign products[i] = lanes_valid[i] ? (act_data[i] * weight_data[i]) : 16'sd0;
        end
    endgenerate

endmodule
