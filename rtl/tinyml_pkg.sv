// =============================================================================
// tinyml_pkg.sv — Shared package for the TinyML Accelerator
//
// PURPOSE:
//   Central home for every compile-time parameter, opcode/activation enum,
//   and the packed `decoded_instr_t` struct.  Import this package in every
//   RTL module with `import tinyml_pkg::*;`.  Do NOT replicate constants
//   anywhere else; change them here and recompile.
//
// CONTENTS:
//   - Top-level architecture parameters (overridable at instantiation)
//   - Opcode and activation enumeration types
//   - decoded_instr_t packed struct — the canonical output of instruction_decoder
//   - Saturation helper function sat8() (pure combinational, synthesisable)
// =============================================================================

package tinyml_pkg;

  // ---------------------------------------------------------------------------
  // Architecture parameters
  // Override these at the top-level instantiation where needed.
  // ---------------------------------------------------------------------------
  parameter int unsigned IMEM_DEPTH_WORDS = 1024;   // 32-bit instruction words
  parameter int unsigned SPAD_DEPTH       = 4096;   // signed INT8 activation elements
  parameter int unsigned PARAM_DEPTH      = 65536;  // signed INT8 elements per parameter bank
  parameter int unsigned OUT_DEPTH        = 4096;   // signed INT8 host-visible output elements
  parameter int unsigned SIMD_WIDTH       = 8;      // products per inner-loop iteration
  parameter int unsigned ACC_WIDTH        = 32;     // signed accumulation width (bits)
  parameter int unsigned REQUANT_SHIFT    = 0;      // arithmetic right shift before INT8 saturation
  parameter int unsigned RELU6_MAX        = 127;    // encoded ReLU6 ceiling

  // ---------------------------------------------------------------------------
  // Opcode encoding  (8-bit, matches sw/compiler/compiler.py)
  // ---------------------------------------------------------------------------
  typedef enum logic [7:0] {
    OP_NOP      = 8'h00,
    OP_LOAD     = 8'h01,
    OP_STORE    = 8'h02,
    OP_DENSE    = 8'h03,
    OP_ACT      = 8'h04,
    OP_END      = 8'h05,
    OP_CONV_CFG = 8'h06,  // Store convolution geometry into config register
    OP_CONV     = 8'h07   // Execute convolution: im2col -> dense multiply
  } opcode_t;

  // ---------------------------------------------------------------------------
  // Activation function encoding  (carried in instr header [15:8])
  // ---------------------------------------------------------------------------
  typedef enum logic [7:0] {
    ACT_NONE    = 8'h00,
    ACT_RELU    = 8'h01,
    ACT_RELU6   = 8'h02,
    ACT_SIGMOID = 8'h03,
    ACT_TANH    = 8'h04
  } activation_t;

  // ---------------------------------------------------------------------------
  // Decoded instruction struct
  //   instruction_decoder drives this; controller_fsm and dense_engine consume it.
  //   Unused operand fields are zero for every opcode that does not define them.
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic        valid;        // 1 = successfully decoded, 0 = decode error
    logic        decode_error; // 1 = illegal opcode, reserved bits, bad activation, etc.
    opcode_t     opcode;       // 8-bit opcode enum
    logic [7:0]  flags;        // header [23:16] — reserved for future extensions
    activation_t activation;   // header [15:8]  — activation type for ACT/DENSE/CONV
    logic signed [7:0] shift;  // header [7:0]   — signed shift amount n for DENSE/CONV

    // LOAD / STORE operands
    logic [15:0] mem_addr;     // LOAD: source host-mem address; STORE: destination
    logic [15:0] sp_addr;      // LOAD: dest scratchpad address; STORE: source

    // DENSE / CONV operands
    logic [15:0] input_addr;
    logic [15:0] weight_addr;
    logic [15:0] output_addr;
    logic [15:0] bias_addr;
    logic [15:0] input_len;     // DENSE input length
    logic [15:0] output_len;    // DENSE output length
    logic [15:0] out_h;         // CONV output height
    logic [15:0] out_w;         // CONV output width
    logic [31:0] m0;            // DENSE/CONV 32-bit requant multiplier M0

    // ACT / shared length operand
    logic [15:0] length;       // LOAD/STORE: byte count; ACT: element count

    // CONV_CFG operands (convolution geometry)
    logic [15:0] in_channels;  // number of input channels
    logic [15:0] out_channels; // number of output channels
    logic [15:0] h_in;         // input feature map height
    logic [15:0] w_in;         // input feature map width
    logic [15:0] kh;           // kernel height
    logic [15:0] kw;           // kernel width
    logic [15:0] stride;       // convolution stride
    logic [15:0] pad;          // zero-padding amount

    // fetch-side bookkeeping (filled by instruction_fetch, forwarded through decoder)
    logic [15:0] next_pc;      // PC value after this instruction's words
  } decoded_instr_t;

  // ---------------------------------------------------------------------------
  // Saturation helper — clamps a signed ACC_WIDTH value to signed INT8 [-128, 127]
  //   TODO: implement the clamping logic using signed comparisons
  // ---------------------------------------------------------------------------
  function automatic logic signed [7:0] sat8(input logic signed [ACC_WIDTH-1:0] val);
    if (val > 32'sd127)
      sat8 = 8'sd127;
    else if (val < -32'sd128)
      sat8 = -8'sd128;
    else
      sat8 = val[7:0];
  endfunction

endpackage
