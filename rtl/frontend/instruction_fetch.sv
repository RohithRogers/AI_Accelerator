// =============================================================================
// instruction_fetch.sv — Variable-Length Instruction Fetch Unit
//
// PURPOSE:
//   Given a starting PC (word address), reads exactly as many 32-bit words as
//   the opcode in the header word demands, assembles them into a raw word
//   bundle, and presents them to the decoder.  It also computes `next_pc` so
//   the controller can advance the program counter in RETIRE without knowing
//   the instruction size itself.
//
//   Word-count table (from ISA spec):
//     NOP  / END  -> 1 word  (header only)
//     LOAD / STORE -> 3 words (header + 2 operand words)
//     ACT          -> 2 words (header + 1 operand word)
//     DENSE        -> 4 words (header + 3 operand words)
//
// STATE MACHINE (internal, simple):
//   IDLE -> (start asserted) -> READ_HDR -> [wait 1 cycle for rd_data]
//        -> PARSE_SIZE -> READ_OPERANDS (0..2 extra words) -> PRESENT
//
// TIMING:
//   instruction_memory has a 1-cycle read latency.  Each word fetch therefore
//   takes one cycle to issue and one cycle to receive.  `instr_valid` is a
//   one-cycle pulse when all words are assembled.
//
// CONNECTIONS:
//   Input  : `fetch_start` from controller_fsm (pulse in FETCH state)
//            `pc_in` from program_counter
//   Memory : drives instruction_memory read port
//   Output : raw word bundle (w0..w3) and `next_pc` to instruction_decoder
//            `instr_valid` pulse to controller_fsm
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module instruction_fetch (
  input  logic        clk,
  input  logic        rst_n,

  // Handshake with controller_fsm
  input  logic        fetch_start,       // controller pulses this to begin a fetch
  output logic        instr_valid,       // one-cycle pulse: raw words are ready
  output logic        fetch_error,       // illegal opcode detected in header

  // Current PC from program_counter
  input  logic [15:0] pc_in,

  // Assembled raw words — passed directly to instruction_decoder
  // w0 is always the header; w1..w4 are operand words (zero when unused)
  output logic [31:0] w0,               // header word (opcode + flags + activation + shift)
  output logic [31:0] w1,               // operand word 1
  output logic [31:0] w2,               // operand word 2
  output logic [31:0] w3,               // operand word 3
  output logic [31:0] w4,               // operand word 4

  // Next PC — forwarded to controller_fsm for retirement
  output logic [15:0] next_pc,

  // Instruction memory read port
  output logic        imem_rd_en,
  output logic [15:0] imem_rd_addr,     // NOTE: sized to IMEM word address, not byte
  input  logic [31:0] imem_rd_data
);

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_WAIT_HDR_MEM,
    ST_WAIT_HDR,
    ST_WAIT_OP,
    ST_CAPTURE_OP
  } state_t;

  state_t state;
  logic [15:0] total_words;
  logic [2:0]  current_op_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= ST_IDLE;
      instr_valid    <= 1'b0;
      fetch_error    <= 1'b0;
      w0             <= 32'h0;
      w1             <= 32'h0;
      w2             <= 32'h0;
      w3             <= 32'h0;
      w4             <= 32'h0;
      next_pc        <= 16'h0;
      imem_rd_en     <= 1'b0;
      imem_rd_addr   <= 16'h0;
      total_words    <= 16'h0;
      current_op_idx <= 3'd0;
    end else begin
      instr_valid <= 1'b0; // Default pulse low
      imem_rd_en  <= 1'b0;

      case (state)
        ST_IDLE: begin
          fetch_error <= 1'b0;
          if (fetch_start) begin
            w0           <= 32'h0;
            w1           <= 32'h0;
            w2           <= 32'h0;
            w3           <= 32'h0;
            w4           <= 32'h0;
            imem_rd_en   <= 1'b1;
            imem_rd_addr <= pc_in;
            state        <= ST_WAIT_HDR_MEM;
          end
        end

        ST_WAIT_HDR_MEM: begin
          state <= ST_WAIT_HDR;
        end

        ST_WAIT_HDR: begin
          // imem_rd_data now contains header word (w0)
          w0 <= imem_rd_data;
          case (imem_rd_data[31:24])
            OP_NOP, OP_END  : total_words <= 16'd1;
            OP_ACT          : total_words <= 16'd2;
            OP_LOAD, OP_STORE: total_words <= 16'd3;
            OP_DENSE        : total_words <= 16'd5;
            OP_CONV_CFG     : total_words <= 16'd5;
            OP_CONV         : total_words <= 16'd5;
            default         : begin
              fetch_error <= 1'b1;
              state       <= ST_IDLE;
            end
          endcase

          if (imem_rd_data[31:24] == OP_NOP   || imem_rd_data[31:24] == OP_LOAD  ||
              imem_rd_data[31:24] == OP_STORE || imem_rd_data[31:24] == OP_DENSE ||
              imem_rd_data[31:24] == OP_ACT   || imem_rd_data[31:24] == OP_END   ||
              imem_rd_data[31:24] == OP_CONV_CFG || imem_rd_data[31:24] == OP_CONV) begin
            logic [15:0] calculated_words;
            case (imem_rd_data[31:24])
              OP_NOP, OP_END   : calculated_words = 16'd1;
              OP_ACT           : calculated_words = 16'd2;
              OP_LOAD, OP_STORE: calculated_words = 16'd3;
              OP_DENSE         : calculated_words = 16'd5;
              OP_CONV_CFG      : calculated_words = 16'd5;
              OP_CONV          : calculated_words = 16'd5;
              default          : calculated_words = 16'd1;
            endcase

            next_pc <= pc_in + calculated_words;

            if (calculated_words == 16'd1) begin
              instr_valid <= 1'b1;
              state       <= ST_IDLE;
            end else begin
              current_op_idx <= 3'd1;
              imem_rd_en     <= 1'b1;
              imem_rd_addr   <= pc_in + 16'd1;
              state          <= ST_WAIT_OP;
            end
          end
        end

        ST_WAIT_OP: begin
          state <= ST_CAPTURE_OP;
        end

        ST_CAPTURE_OP: begin
          case (current_op_idx)
            3'd1: w1 <= imem_rd_data;
            3'd2: w2 <= imem_rd_data;
            3'd3: w3 <= imem_rd_data;
            3'd4: w4 <= imem_rd_data;
            default: ;
          endcase

          if (current_op_idx + 1'b1 == total_words[2:0]) begin
            instr_valid <= 1'b1;
            state       <= ST_IDLE;
          end else begin
            current_op_idx <= current_op_idx + 1'b1;
            imem_rd_en     <= 1'b1;
            imem_rd_addr   <= pc_in + {13'd0, current_op_idx + 1'b1};
            state          <= ST_WAIT_OP;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
