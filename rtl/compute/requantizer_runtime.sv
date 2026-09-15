// Runtime Q31 requantization matching sw/compiler/compiler.py.
module requantizer_runtime #(parameter int ACC_WIDTH = 32) (
  input logic signed [ACC_WIDTH-1:0] acc_in,
  input logic [31:0] m0,
  input logic signed [7:0] shift_n,
  output logic signed [7:0] val_out
);
  // Use a function so Icarus can handle variable shifts cleanly
  function automatic logic signed [63:0] do_scale(
    input logic signed [ACC_WIDTH-1:0] acc,
    input logic [31:0]                 m,
    input logic signed [7:0]           sh
  );
    logic signed [63:0] prod;
    int total_sh;
    begin
      prod     = acc * $signed({1'b0, m[30:0]});
      total_sh = 31 + int'(sh);
      if (m == 0)
        do_scale = 64'sd0;
      // Match the compiler/golden model: add half an LSB before the
      // arithmetic shift (including for negative values).
      else if (total_sh > 0)
        do_scale = (prod + (64'sd1 <<< (total_sh - 1))) >>> total_sh;
      else if (total_sh == 0)
        do_scale = prod;
      else
        do_scale = prod <<< (-total_sh);
    end
  endfunction

  logic signed [63:0] scaled;
  assign scaled  = do_scale(acc_in, m0, shift_n);
  assign val_out = (scaled >  127)  ? 8'sd127  :
                   (scaled < -128)  ? -8'sd128 :
                                      scaled[7:0];
endmodule
