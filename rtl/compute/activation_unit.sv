// ============================================================================
// File: activation_unit.sv
// Description: Evaluates non-linear activation functions on signed 8-bit inputs.
//              Supports NONE, ReLU, ReLU6, Sigmoid, and Tanh.
//              Sigmoid/Tanh are implemented via precomputed lookup tables (LUTs).
// ============================================================================

import tinyml_pkg::*;

module activation_unit #(
    parameter int RELU6_MAX = 127
) (
    input  logic [7:0]          act_type, // Activation function selector byte
    input  logic signed [7:0]   data_in,  // Scaled input activation byte
    output logic signed [7:0]   data_out  // Activated output activation byte
);

    // Precomputed Sigmoid LUT mapping data_in [-128, 127] -> y [0, 127]
    // Assumes input scale = 0.0625, output scale = 1/127
    localparam logic signed [7:0] SIGMOID_LUT [256] = '{
      8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0,
      8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0,
      8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd0, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1,
      8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd2, 8'sd2, 8'sd2, 8'sd2, 8'sd2, 8'sd2,
      8'sd2, 8'sd2, 8'sd3, 8'sd3, 8'sd3, 8'sd3, 8'sd3, 8'sd4, 8'sd4, 8'sd4, 8'sd4, 8'sd4, 8'sd5, 8'sd5, 8'sd5, 8'sd6,
      8'sd6, 8'sd6, 8'sd7, 8'sd7, 8'sd8, 8'sd8, 8'sd9, 8'sd9, 8'sd10, 8'sd10, 8'sd11, 8'sd11, 8'sd12, 8'sd13, 8'sd14, 8'sd14,
      8'sd15, 8'sd16, 8'sd17, 8'sd18, 8'sd19, 8'sd20, 8'sd21, 8'sd22, 8'sd23, 8'sd24, 8'sd26, 8'sd27, 8'sd28, 8'sd30, 8'sd31, 8'sd33,
      8'sd34, 8'sd36, 8'sd37, 8'sd39, 8'sd41, 8'sd42, 8'sd44, 8'sd46, 8'sd48, 8'sd50, 8'sd52, 8'sd54, 8'sd56, 8'sd58, 8'sd60, 8'sd62,
      8'sd64, 8'sd65, 8'sd67, 8'sd69, 8'sd71, 8'sd73, 8'sd75, 8'sd77, 8'sd79, 8'sd81, 8'sd83, 8'sd85, 8'sd86, 8'sd88, 8'sd90, 8'sd91,
      8'sd93, 8'sd94, 8'sd96, 8'sd97, 8'sd99, 8'sd100, 8'sd101, 8'sd103, 8'sd104, 8'sd105, 8'sd106, 8'sd107, 8'sd108, 8'sd109, 8'sd110, 8'sd111,
      8'sd112, 8'sd113, 8'sd113, 8'sd114, 8'sd115, 8'sd116, 8'sd116, 8'sd117, 8'sd117, 8'sd118, 8'sd118, 8'sd119, 8'sd119, 8'sd120, 8'sd120, 8'sd121,
      8'sd121, 8'sd121, 8'sd122, 8'sd122, 8'sd122, 8'sd123, 8'sd123, 8'sd123, 8'sd123, 8'sd123, 8'sd124, 8'sd124, 8'sd124, 8'sd124, 8'sd124, 8'sd125,
      8'sd125, 8'sd125, 8'sd125, 8'sd125, 8'sd125, 8'sd125, 8'sd125, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126,
      8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127,
      8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127,
      8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127
    };

    // Precomputed Tanh LUT mapping data_in [-128, 127] -> y [-128, 127]
    // Assumes input scale = 0.0625, output scale = 1/127
    localparam logic signed [7:0] TANH_LUT [256] = '{
      -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127,
      -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127,
      -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127,
      -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127,
      -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd127, -8'sd126,
      -8'sd126, -8'sd126, -8'sd126, -8'sd126, -8'sd126, -8'sd126, -8'sd126, -8'sd125, -8'sd125, -8'sd125, -8'sd125, -8'sd125, -8'sd124, -8'sd124, -8'sd123, -8'sd123,
      -8'sd122, -8'sd122, -8'sd121, -8'sd120, -8'sd120, -8'sd119, -8'sd118, -8'sd116, -8'sd115, -8'sd113, -8'sd112, -8'sd110, -8'sd108, -8'sd105, -8'sd103, -8'sd100,
      -8'sd97, -8'sd93, -8'sd89, -8'sd85, -8'sd81, -8'sd76, -8'sd70, -8'sd65, -8'sd59, -8'sd52, -8'sd46, -8'sd38, -8'sd31, -8'sd24, -8'sd16, -8'sd8,
      8'sd0, 8'sd8, 8'sd16, 8'sd24, 8'sd31, 8'sd38, 8'sd46, 8'sd52, 8'sd59, 8'sd65, 8'sd70, 8'sd76, 8'sd81, 8'sd85, 8'sd89, 8'sd93,
      8'sd97, 8'sd100, 8'sd103, 8'sd105, 8'sd108, 8'sd110, 8'sd112, 8'sd113, 8'sd115, 8'sd116, 8'sd118, 8'sd119, 8'sd120, 8'sd120, 8'sd121, 8'sd122,
      8'sd122, 8'sd123, 8'sd123, 8'sd124, 8'sd124, 8'sd125, 8'sd125, 8'sd125, 8'sd125, 8'sd125, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126, 8'sd126,
      8'sd126, 8'sd126, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127,
      8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127,
      8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127,
      8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127,
      8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127, 8'sd127
    };

    logic [7:0] index;
    assign index = data_in + 8'd128; // Offset to map [-128, 127] to [0, 255]

    always_comb begin
        case (act_type)
            ACT_NONE: begin
                data_out = data_in; // Pass-through
            end
            
            ACT_RELU: begin
                // Clamps negative inputs to 0
                data_out = (data_in < 8'sd0) ? 8'sd0 : data_in;
            end
            
            ACT_RELU6: begin
                // Clamps inputs to the range [0, RELU6_MAX]
                if (data_in < 8'sd0)
                    data_out = 8'sd0;
                else if (data_in > RELU6_MAX[7:0])
                    data_out = RELU6_MAX[7:0];
                else
                    data_out = data_in;
            end

            ACT_SIGMOID: begin
                data_out = SIGMOID_LUT[index];
            end

            ACT_TANH: begin
                data_out = TANH_LUT[index];
            end
            
            default: begin
                data_out = data_in; // Default fallback
            end
        endcase
    end

endmodule
