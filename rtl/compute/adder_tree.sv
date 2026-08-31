// ============================================================================
// File: adder_tree.sv
// Description: Pairwise reduction adder tree. Combines 4 signed 16-bit 
//              products into a single 32-bit signed accumulator increment.
// ============================================================================

import tinyml_pkg::*;

module adder_tree #(
    parameter int SIMD_WIDTH = 8,  // Number of elements to sum (W=4)
    parameter int ACC_WIDTH  = 32  // Wider width to prevent mathematical overflow
) (
    input  logic signed [15:0]      products[SIMD_WIDTH], // Products from multipliers
    output logic signed [ACC_WIDTH-1:0] sum_out           // Final 32-bit signed sum
);

    always_comb begin
        logic signed [ACC_WIDTH-1:0] temp_sum;
        temp_sum = '0; // Initialize sum to zero
        
        // Sum each lane's product sequentially
        for (int i = 0; i < SIMD_WIDTH; i++) begin
            // Sign-extend the 16-bit product to 32 bits before adding to prevent overflow.
            // Sign-extension replicates the most significant bit (sign bit, products[i][15]) 16 times.
            temp_sum = temp_sum + $signed({{16{products[i][15]}}, products[i]});
        end
        
        sum_out = temp_sum;
    end

endmodule
