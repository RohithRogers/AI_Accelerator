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
    input  logic signed [7:0]     act_data   [SIMD_WIDTH], 
    input  logic signed [7:0]     weight_data[SIMD_WIDTH], 
    
    // Lane mask to selectively disable lanes for non-multiple-of-4 vector lengths
    input  logic [SIMD_WIDTH-1:0] lanes_valid, 
    
    // Outputs: 4 signed 16-bit multiplication products
    output logic signed [15:0]    products   [SIMD_WIDTH]  
);

    // Combinational block executing multiplications
    always_comb begin
        for (int i = 0; i < SIMD_WIDTH; i++) begin
            // If the lane is marked active by the loader, perform signed multiplication
            if (lanes_valid[i]) begin
                // signed 8-bit * signed 8-bit = signed 16-bit result
                products[i] = act_data[i] * weight_data[i];
            end else begin
                // If lane is inactive (padded tail element), output 0 to protect the sum
                products[i] = 16'sd0;
            end
        end
    end

endmodule
