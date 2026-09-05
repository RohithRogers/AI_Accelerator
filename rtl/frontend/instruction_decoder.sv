// =============================================================================
// instruction_decoder.sv — Pure Combinational Instruction Decoder
//
// PURPOSE:
//   Takes the raw 32-bit word bundle from instruction_fetch and extracts every
//   field into a `decoded_instr_t` struct.  This module is PURELY combinational:
//   no registers, no clock port, no state.  It mirrors `decode_stream()` in
//   sw/compiler/test_roundtrip.py exactly — bit fields, endianness, and all.
//
// DECODING RULES (from ISA spec):
//   Header word  w0[31:24] = opcode
//                w0[23:16] = flags   (reserved; must be 0x00 in V1)
//                w0[15:8]  = activation
//                w0[7:0]   = reserved (must be 0x00)
//
//   LOAD  : w1[31:16]=mem_addr, w1[15:0]=sp_addr,  w2[31:16]=length, w2[15:0]=reserved(0)
//   STORE : w1[31:16]=sp_addr,  w1[15:0]=mem_addr, w2[31:16]=length, w2[15:0]=reserved(0)
//   DENSE : w1[31:16]=input_addr, w1[15:0]=weight_addr,
//            w2[31:16]=output_addr, w2[15:0]=bias_addr,
//            w3[31:16]=input_len,  w3[15:0]=output_len
//   ACT   : w1[31:16]=addr, w1[15:0]=length
//   NOP/END: no operands; all operand words must be zero
//
// ERROR CONDITIONS (raise decode_error, clear valid):
//   - Unknown opcode (not in {OP_NOP, OP_LOAD, OP_STORE, OP_DENSE, OP_ACT, OP_END})
//   - flags field (w0[23:16]) != 8'h00
//   - reserved byte (w0[7:0]) != 8'h00
//   - activation field not in {ACT_NONE, ACT_RELU, ACT_RELU6}
//   - LOAD/STORE: w2[15:0] (reserved half) != 0
//   - NOP/END: any operand word w1/w2/w3 != 0
//
// OUTPUT:
//   A single `decoded_instr_t` struct (from tinyml_pkg).  All unused fields
//   are driven to zero.  `next_pc` is driven to zero here; instruction_fetch
//   fills it before the decoder is consulted.
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module instruction_decoder (
  // Raw word bundle from instruction_fetch (combinational inputs)
  input  logic [31:0] w0,               // header
  input  logic [31:0] w1,               // operand word 1 (zero if unused)
  input  logic [31:0] w2,               // operand word 2 (zero if unused)
  input  logic [31:0] w3,               // operand word 3 (zero if unused)
  input  logic [31:0] w4,               // operand word 4 (zero if unused)

  // Decoded instruction struct output (fully combinational, type from tinyml_pkg)
  output decoded_instr_t instr_out
);

  // ---------------------------------------------------------------------------
  // Internal intermediate signals — declared before use
  // ---------------------------------------------------------------------------
  logic        instr_decoder_error;
  opcode_t     instr_type;              // decoded opcode enum (from package)

  // ---------------------------------------------------------------------------
  // 1. Opcode type decode (pure combinational)
  //    Maps the raw 8-bit opcode field to the package opcode_t enum.
  //    Unknown opcodes fall to OP_NOP so downstream fields extract safely,
  //    but instr_decoder_error will be set, so no instruction fires.
  // ---------------------------------------------------------------------------
  always @* begin
    case (w0[31:24])
      OP_NOP      : instr_type = OP_NOP;
      OP_LOAD     : instr_type = OP_LOAD;
      OP_STORE    : instr_type = OP_STORE;
      OP_DENSE    : instr_type = OP_DENSE;
      OP_ACT      : instr_type = OP_ACT;
      OP_END      : instr_type = OP_END;
      OP_CONV_CFG : instr_type = OP_CONV_CFG;
      OP_CONV     : instr_type = OP_CONV;
      default     : instr_type = OP_NOP;   // unknown — error will be flagged below
    endcase
  end

  // ---------------------------------------------------------------------------
  // 2. Error detection (combinational)
  //
  //    Reserved-field checks per ISA (only fields the spec calls reserved):
  //      Header  : flags (w0[23:16]) must be 0x00
  //                shift (w0[7:0]) must be 0x00 for NOP, LOAD, STORE, ACT, END.
  //      LOAD    : w2[15:0] must be 0  (spec: W2={length[31:16], reserved[15:0]})
  //      STORE   : w2[15:0] must be 0  (same layout as LOAD)
  //      DENSE   : no reserved halves in W1–W3; all fields are operands
  //      ACT     : no reserved halves in W1; both halves are addr/length
  //      NOP/END : w1, w2, w3, w4 must all be zero
  // ---------------------------------------------------------------------------
  assign instr_decoder_error =
    // Unknown opcode
    !(w0[31:24] == OP_NOP   || w0[31:24] == OP_LOAD  ||
      w0[31:24] == OP_STORE || w0[31:24] == OP_DENSE ||
      w0[31:24] == OP_ACT   || w0[31:24] == OP_END   ||
      w0[31:24] == OP_CONV_CFG || w0[31:24] == OP_CONV) ||
    // Header reserved fields
    (w0[23:16] != 8'h00) ||
    (instr_type != OP_DENSE && instr_type != OP_CONV && w0[7:0] != 8'h00) ||
    // Invalid activation encoding
    !(w0[15:8] == ACT_NONE || w0[15:8] == ACT_RELU || w0[15:8] == ACT_RELU6 ||
      w0[15:8] == ACT_SIGMOID || w0[15:8] == ACT_TANH) ||
    // LOAD/STORE: w2 lower half must be zero, w3 and w4 must be zero
    (instr_type == OP_LOAD  && (w2[15:0] != 16'h0000 || w3 != 32'h0 || w4 != 32'h0)) ||
    (instr_type == OP_STORE && (w2[15:0] != 16'h0000 || w3 != 32'h0 || w4 != 32'h0)) ||
    // ACT: w2, w3, w4 must be zero
    (instr_type == OP_ACT   && (w2 != 32'h0 || w3 != 32'h0 || w4 != 32'h0)) ||
    // NOP: no operand words expected
    (instr_type == OP_NOP   && (w1 != 32'h0 || w2 != 32'h0 || w3 != 32'h0 || w4 != 32'h0)) ||
    // END: no operand words expected
    (instr_type == OP_END   && (w1 != 32'h0 || w2 != 32'h0 || w3 != 32'h0 || w4 != 32'h0));

  // ---------------------------------------------------------------------------
  // 3. Output struct field assignments
  //
  //    Fields in decoded_instr_t (from tinyml_pkg):
  //      valid, decode_error, opcode, flags, activation, shift,
  //      mem_addr, sp_addr,
  //      input_addr, weight_addr, output_addr, bias_addr,
  //      input_len, output_len, out_h, out_w, m0,
  //      length,
  //      in_channels, out_channels, h_in, w_in, kh, kw, stride, pad,
  //      next_pc
  // ---------------------------------------------------------------------------

  // Handshake fields
  assign instr_out.valid        = ~instr_decoder_error;
  assign instr_out.decode_error =  instr_decoder_error;

  // Header fields (always extracted, even on error, for diagnostics)
  assign instr_out.opcode       = opcode_t'(w0[31:24]);
  assign instr_out.flags        = w0[23:16];
  assign instr_out.activation   = activation_t'(w0[15:8]);
  assign instr_out.shift        = (instr_type == OP_DENSE || instr_type == OP_CONV) ? w0[7:0] : 8'h00;

  // LOAD / STORE / ACT operands
  assign instr_out.mem_addr     = (instr_type == OP_LOAD)  ? w1[31:16] :
                                  (instr_type == OP_STORE) ? w1[15:0]  : 16'h0000;
  assign instr_out.sp_addr      = (instr_type == OP_LOAD)  ? w1[15:0]  :
                                  (instr_type == OP_STORE || instr_type == OP_ACT) ? w1[31:16] : 16'h0000;

  // Shared length field (LOAD/STORE byte count from w2[31:16]; ACT element count from w1[15:0])
  assign instr_out.length       = (instr_type == OP_LOAD  || instr_type == OP_STORE) ? w2[31:16] :
                                  (instr_type == OP_ACT)                              ? w1[15:0]  :
                                                                                        16'h0000;

  // DENSE / CONV operands
  assign instr_out.input_addr   = (instr_type == OP_DENSE || instr_type == OP_CONV) ? w1[31:16] : 16'h0000;
  assign instr_out.weight_addr  = (instr_type == OP_DENSE || instr_type == OP_CONV) ? w1[15:0]  : 16'h0000;
  assign instr_out.output_addr  = (instr_type == OP_DENSE || instr_type == OP_CONV) ? w2[31:16] : 16'h0000;
  assign instr_out.bias_addr    = (instr_type == OP_DENSE || instr_type == OP_CONV) ? w2[15:0]  : 16'h0000;
  assign instr_out.input_len    = (instr_type == OP_DENSE) ? w3[31:16] : 16'h0000;
  assign instr_out.output_len   = (instr_type == OP_DENSE) ? w3[15:0]  : 16'h0000;
  assign instr_out.out_h        = (instr_type == OP_CONV)  ? w3[31:16] : 16'h0000;
  assign instr_out.out_w        = (instr_type == OP_CONV)  ? w3[15:0]  : 16'h0000;
  assign instr_out.m0           = (instr_type == OP_DENSE || instr_type == OP_CONV) ? w4 : 32'h0;

  // CONV_CFG operands
  assign instr_out.in_channels  = (instr_type == OP_CONV_CFG) ? w1[31:16] : 16'h0000;
  assign instr_out.out_channels = (instr_type == OP_CONV_CFG) ? w1[15:0]  : 16'h0000;
  assign instr_out.h_in         = (instr_type == OP_CONV_CFG) ? w2[31:16] : 16'h0000;
  assign instr_out.w_in         = (instr_type == OP_CONV_CFG) ? w2[15:0]  : 16'h0000;
  assign instr_out.kh           = (instr_type == OP_CONV_CFG) ? w3[31:16] : 16'h0000;
  assign instr_out.kw           = (instr_type == OP_CONV_CFG) ? w3[15:0]  : 16'h0000;
  assign instr_out.stride       = (instr_type == OP_CONV_CFG) ? w4[31:16] : 16'h0000;
  assign instr_out.pad          = (instr_type == OP_CONV_CFG) ? w4[15:0]  : 16'h0000;

  // next_pc: this decoder does not compute it; instruction_fetch fills it.
  // Drive zero so the struct is fully assigned (avoids X propagation in sim).
  assign instr_out.next_pc      = 16'h0000;

endmodule
