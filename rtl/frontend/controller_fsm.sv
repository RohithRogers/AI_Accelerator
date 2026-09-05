// =============================================================================
// controller_fsm.sv — Top-Level Sequencing FSM
//
// PURPOSE:
//   Owns program sequencing: it is the only block that may advance the PC,
//   dispatch instructions to executors, and pulse the top-level done/error
//   signals.  The dense_engine and memory copy loops are *sub-executors*;
//   the controller only starts them and waits for their `done`.
//
// STATE SEQUENCE (registered one-hot or binary — your choice):
//
//   IDLE       : waits for top-level `start` pulse.
//   FETCH      : asserts fetch_start, waits for instr_valid from instruction_fetch.
//   DECODE     : latches decoded_instr_t from instruction_decoder (one cycle).
//   DISPATCH   : checks opcode and bounds; routes to the correct executor state.
//                Goes to ERROR on decode_error, bad opcode, or failed bounds check.
//   LOAD_COPY  : iterates a 16-bit `remaining` counter, copying bytes one-per-cycle
//                from host-data memory to scratchpad.  Increments src/dst addresses.
//   STORE_COPY : same as LOAD_COPY but scratchpad -> output memory direction.
//   DENSE_START: asserts dense_engine `start` for one cycle with all operands stable.
//   DENSE_WAIT : waits for dense_engine `done` or `error`.
//   ACT_LOOP   : iterates activation in-place on scratchpad (or delegates to activation_unit).
//   NOP        : one idle cycle, then RETIRE.
//   RETIRE     : asserts `pc_write` with the latched `next_pc`; then goes back to FETCH.
//   DONE       : pulses top-level `done` for one cycle; then goes to IDLE.
//   ERROR      : asserts top-level `error` + `error_code`; holds until rst_n or policy allows restart.
//
// KEY RULES:
//   - PC advances ONLY in RETIRE.
//   - `done` is exactly one clock wide; `busy` is high from accepted start through RETIRE.
//   - END goes DISPATCH -> END_STATE -> DONE -> IDLE (no RETIRE, PC does not advance).
//   - Any executor error routes to ERROR with a distinctive error_code.
//
// CONNECTIONS (see implementation.md for full width table):
//   Top-level  : start, busy, done, error, error_code
//   Fetch unit : fetch_start, instr_valid, fetch_error, w0..w3, next_pc_from_fetch
//   Decoder    : instr_out (decoded_instr_t) — purely combinational, always valid
//   PC         : pc_write, pc_next
//   Dense eng  : dense_start, dense_busy, dense_done, dense_error
//                + input_addr, weight_addr, output_addr, bias_addr,
//                  input_len, output_len, activation (held stable until done)
//   Memories   : scratchpad rd/wr port, output_mem wr port, host_mem rd port
//                (for LOAD_COPY / STORE_COPY byte loops)
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module controller_fsm (
  input  logic clk,
  input  logic rst_n,

  // ---- Top-level execution interface ----------------------------------------
  input  logic        start,             // sampled only while in IDLE
  output logic        busy,              // high from FETCH through RETIRE
  output logic        done,              // one-cycle pulse in DONE state
  output logic        error,             // asserted in ERROR state
  output logic [7:0]  error_code,        // distinguishes error source

  // ---- Instruction fetch handshake ------------------------------------------
  output logic        fetch_start,       // pulse to begin a fetch
  input  logic        instr_valid,       // one-cycle pulse: words assembled
  input  logic        fetch_error,       // illegal header opcode from fetch unit
  input  logic [31:0] w0, w1, w2, w3,   // raw word bundle for decoder

  // ---- Decoded instruction (combinational from instruction_decoder) ----------
  input  decoded_instr_t instr,          // always-valid combinational decode of w0..w3

  // ---- Program counter control ----------------------------------------------
  output logic        pc_write,          // strobe: commit next_pc to PC register
  output logic [15:0] pc_next,           // next_pc value (latched from fetch)
  input  logic [15:0] pc_out,            // current PC (from program_counter)

  // ---- Dense engine command interface ---------------------------------------
  output logic        dense_start,       // one-cycle start pulse
  input  logic        dense_busy,
  input  logic        dense_done,        // one-cycle done pulse
  input  logic        dense_error,
  output logic [15:0] dense_input_addr,
  output logic [15:0] dense_weight_addr,
  output logic [15:0] dense_output_addr,
  output logic [15:0] dense_bias_addr,
  output logic [15:0] dense_input_len,
  output logic [15:0] dense_output_len,
  output logic [7:0]  dense_activation,
  output logic [31:0] dense_requant_m0,
  output logic signed [7:0] dense_requant_shift,

  // ---- Scratchpad read/write (LOAD_COPY dst / ACT_LOOP / STORE_COPY src) ----
  output logic        spad_wr_en,
  output logic [15:0] spad_wr_addr,
  output logic signed [7:0] spad_wr_data,
  output logic        spad_rd_en,
  output logic [15:0] spad_rd_addr,
  input  logic signed [7:0] spad_rd_data,

  // ---- Host-data memory read (LOAD_COPY source) -----------------------------
  output logic        hmem_rd_en,
  output logic [15:0] hmem_rd_addr,
  input  logic signed [7:0] hmem_rd_data,

  // ---- Output memory write (STORE_COPY destination) -------------------------
  output logic        omem_wr_en,
  output logic [15:0] omem_wr_addr,
  output logic signed [7:0] omem_wr_data
);

  // ---------------------------------------------------------------------------
  // FSM state encoding (binary — 4 bits for 14 states)
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    ST_IDLE        = 4'd0,
    ST_FETCH       = 4'd1,
    ST_DECODE      = 4'd2,
    ST_DISPATCH    = 4'd3,
    ST_LOAD_COPY   = 4'd4,
    ST_STORE_COPY  = 4'd5,
    ST_DENSE_START = 4'd6,
    ST_DENSE_WAIT  = 4'd7,
    ST_ACT_LOOP    = 4'd8,
    ST_NOP_ST      = 4'd9,
    ST_RETIRE      = 4'd10,
    ST_DONE_ST     = 4'd11,
    ST_END_ST      = 4'd12,
    ST_ERROR_ST    = 4'd13
  } state_t;

  state_t current_state, next_state;

  // ---------------------------------------------------------------------------
  // Working registers
  // ---------------------------------------------------------------------------
  decoded_instr_t latched_instr;   // decoded instruction captured in DECODE
  logic [15:0]    next_pc_latch;   // next_pc captured when instr_valid pulses

  // LOAD_COPY / STORE_COPY / ACT_LOOP pipelined iteration registers
  logic [15:0] copy_src_addr;      // base source address
  logic [15:0] copy_dst_addr;      // base destination address
  logic [15:0] copy_length;        // total bytes/elements to copy
  logic [15:0] copy_read_count;    // number of read requests issued (0..length)
  logic [15:0] copy_write_count;   // number of write responses committed (0..length)
  logic [15:0] pipe_dst_addr;      // latched destination address for 1-cycle latency write
  logic        pipe_valid;         // indicates read data is valid on this cycle
  logic        fetch_req;          // 1-cycle strobe to start fetch after PC commits

  // ACT_LOOP iteration registers
  logic [15:0] act_addr;           // current element base address

  // ---------------------------------------------------------------------------
  // Error code constants
  // ---------------------------------------------------------------------------
  localparam logic [7:0] ERR_FETCH_ILLEGAL = 8'h01;
  localparam logic [7:0] ERR_DECODE        = 8'h02;
  localparam logic [7:0] ERR_DENSE_EXEC    = 8'h03;
  localparam logic [7:0] ERR_BAD_OPCODE    = 8'h04;

  // ---------------------------------------------------------------------------
  // Sequential block: state register + working registers
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      current_state    <= ST_IDLE;
      latched_instr    <= '0;
      next_pc_latch    <= 16'h0000;
      copy_src_addr    <= 16'h0000;
      copy_dst_addr    <= 16'h0000;
      copy_length      <= 16'h0000;
      copy_read_count  <= 16'h0000;
      copy_write_count <= 16'h0000;
      pipe_dst_addr    <= 16'h0000;
      pipe_valid       <= 1'b0;
      fetch_req        <= 1'b0;
      act_addr         <= 16'h0000;
    end else begin
      current_state <= next_state;

      // ---- Manage 1-cycle fetch request strobe -----------------------------
      if (current_state == ST_IDLE && start)
        fetch_req <= 1'b1;
      else if (current_state == ST_RETIRE)
        fetch_req <= 1'b1;
      else if (fetch_req)
        fetch_req <= 1'b0;

      // ---- FETCH: capture next_pc when instruction words arrive ------------
      if (current_state == ST_FETCH && instr_valid)
        next_pc_latch <= instr.next_pc;

      // ---- DECODE: latch the fully-decoded instruction for one cycle -------
      if (current_state == ST_DECODE)
        latched_instr <= instr;

      // ---- DISPATCH: initialise copy pointers from latched instruction -----
      if (current_state == ST_DISPATCH) begin
        copy_read_count  <= 16'h0000;
        copy_write_count <= 16'h0000;
        pipe_valid       <= 1'b0;
        pipe_dst_addr    <= 16'h0000;

        case (latched_instr.opcode)
          OP_LOAD: begin
            copy_src_addr <= latched_instr.mem_addr;
            copy_dst_addr <= latched_instr.sp_addr;
            copy_length   <= latched_instr.length;
          end
          OP_STORE: begin
            copy_src_addr <= latched_instr.sp_addr;
            copy_dst_addr <= latched_instr.mem_addr;
            copy_length   <= latched_instr.length;
          end
          OP_ACT: begin
            act_addr      <= latched_instr.sp_addr;
            copy_dst_addr <= latched_instr.sp_addr;
            copy_length   <= latched_instr.length;
          end
          default: ;
        endcase
      end

      // ---- LOAD_COPY / STORE_COPY / ACT_LOOP pipelined sequencing ---------
      if (current_state == ST_LOAD_COPY || current_state == ST_STORE_COPY || current_state == ST_ACT_LOOP) begin
        if (copy_read_count < copy_length) begin
          copy_read_count <= copy_read_count + 16'd1;
          pipe_dst_addr   <= copy_dst_addr + copy_read_count;
          pipe_valid      <= 1'b1;
        end else begin
          pipe_valid      <= 1'b0;
        end

        if (pipe_valid) begin
          copy_write_count <= copy_write_count + 16'd1;
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Combinational block: next-state logic + output driving
  // ---------------------------------------------------------------------------
  always_comb begin
    // ---- Default outputs (safe idle values) --------------------------------
    next_state          = current_state;
    busy                = 1'b0;
    done                = 1'b0;
    error               = 1'b0;
    error_code          = 8'h00;
    fetch_start         = 1'b0;
    pc_write            = 1'b0;
    pc_next             = next_pc_latch;
    dense_start         = 1'b0;
    dense_input_addr    = latched_instr.input_addr;
    dense_weight_addr   = latched_instr.weight_addr;
    dense_output_addr   = latched_instr.output_addr;
    dense_bias_addr     = latched_instr.bias_addr;
    dense_input_len     = latched_instr.input_len;
    dense_output_len    = latched_instr.output_len;
    dense_activation    = 8'(latched_instr.activation);
    dense_requant_m0    = latched_instr.m0;
    dense_requant_shift = latched_instr.shift;
    spad_wr_en          = 1'b0;
    spad_wr_addr        = 16'h0000;
    spad_wr_data        = 8'sd0;
    spad_rd_en          = 1'b0;
    spad_rd_addr        = 16'h0000;
    hmem_rd_en          = 1'b0;
    hmem_rd_addr        = 16'h0000;
    omem_wr_en          = 1'b0;
    omem_wr_addr        = 16'h0000;
    omem_wr_data        = 8'sd0;

    case (current_state)
      // ---------------------------------------------------------------
      ST_IDLE: begin
        busy = 1'b0;
        if (start)
          next_state = ST_FETCH;
      end

      // ---------------------------------------------------------------
      ST_FETCH: begin
        busy        = 1'b1;
        fetch_start = fetch_req;
        if (fetch_error)
          next_state = ST_ERROR_ST;
        else if (instr_valid)
          next_state = ST_DECODE;
        else
          next_state = ST_FETCH;     // hold until instr_valid
      end

      // ---------------------------------------------------------------
      // DECODE: one cycle to let the latch capture latched_instr (done in FF)
      ST_DECODE: begin
        busy       = 1'b1;
        next_state = ST_DISPATCH;
      end

      // ---------------------------------------------------------------
      ST_DISPATCH: begin
        busy = 1'b1;
        if (latched_instr.decode_error) begin
          next_state = ST_ERROR_ST;
        end else begin
          case (latched_instr.opcode)
            OP_NOP  : next_state = ST_NOP_ST;
            OP_LOAD : next_state = ST_LOAD_COPY;
            OP_STORE: next_state = ST_STORE_COPY;
            OP_DENSE: next_state = ST_DENSE_START;
            OP_ACT  : next_state = ST_ACT_LOOP;
            OP_END  : next_state = ST_END_ST;
            default : next_state = ST_ERROR_ST;
          endcase
        end
      end

      // ---------------------------------------------------------------
      // LOAD_COPY: pipelined byte copy from host memory to scratchpad
      ST_LOAD_COPY: begin
        busy = 1'b1;
        if (copy_length == 16'h0000 || (pipe_valid && (copy_write_count + 16'd1 == copy_length))) begin
          if (pipe_valid) begin
            spad_wr_en   = 1'b1;
            spad_wr_addr = pipe_dst_addr;
            spad_wr_data = hmem_rd_data;
          end
          next_state = ST_RETIRE;
        end else begin
          if (copy_read_count < copy_length) begin
            hmem_rd_en   = 1'b1;
            hmem_rd_addr = copy_src_addr + copy_read_count;
          end
          if (pipe_valid) begin
            spad_wr_en   = 1'b1;
            spad_wr_addr = pipe_dst_addr;
            spad_wr_data = hmem_rd_data;
          end
          next_state = ST_LOAD_COPY;
        end
      end

      // ---------------------------------------------------------------
      // STORE_COPY: pipelined byte copy from scratchpad to output memory
      ST_STORE_COPY: begin
        busy = 1'b1;
        if (copy_length == 16'h0000 || (pipe_valid && (copy_write_count + 16'd1 == copy_length))) begin
          if (pipe_valid) begin
            omem_wr_en   = 1'b1;
            omem_wr_addr = pipe_dst_addr;
            omem_wr_data = spad_rd_data;
          end
          next_state = ST_RETIRE;
        end else begin
          if (copy_read_count < copy_length) begin
            spad_rd_en   = 1'b1;
            spad_rd_addr = copy_src_addr + copy_read_count;
          end
          if (pipe_valid) begin
            omem_wr_en   = 1'b1;
            omem_wr_addr = pipe_dst_addr;
            omem_wr_data = spad_rd_data;
          end
          next_state = ST_STORE_COPY;
        end
      end

      // ---------------------------------------------------------------
      ST_DENSE_START: begin
        busy        = 1'b1;
        dense_start = 1'b1;           // one-cycle start pulse
        next_state  = ST_DENSE_WAIT;
      end

      // ---------------------------------------------------------------
      ST_DENSE_WAIT: begin
        busy = 1'b1;
        if (dense_error)
          next_state = ST_ERROR_ST;
        else if (dense_done)
          next_state = ST_RETIRE;
        else
          next_state = ST_DENSE_WAIT;
      end

      // ---------------------------------------------------------------
      // ACT_LOOP: pipelined in-place ReLU / ReLU6 on scratchpad
      ST_ACT_LOOP: begin
        busy = 1'b1;
        if (copy_length == 16'h0000 || (pipe_valid && (copy_write_count + 16'd1 == copy_length))) begin
          if (pipe_valid) begin
            logic signed [7:0] act_in;
            logic signed [7:0] act_out;
            act_in = spad_rd_data;
            case (latched_instr.activation)
              ACT_RELU : act_out = (act_in < 8'sd0) ? 8'sd0 : act_in;
              ACT_RELU6: act_out = (act_in < 8'sd0) ? 8'sd0 :
                                   (act_in > 8'(RELU6_MAX)) ? 8'(RELU6_MAX) : act_in;
              default  : act_out = act_in;  // ACT_NONE
            endcase
            spad_wr_en   = 1'b1;
            spad_wr_addr = pipe_dst_addr;
            spad_wr_data = act_out;
          end
          next_state = ST_RETIRE;
        end else begin
          if (copy_read_count < copy_length) begin
            spad_rd_en   = 1'b1;
            spad_rd_addr = act_addr + copy_read_count;
          end
          if (pipe_valid) begin
            logic signed [7:0] act_in;
            logic signed [7:0] act_out;
            act_in = spad_rd_data;
            case (latched_instr.activation)
              ACT_RELU : act_out = (act_in < 8'sd0) ? 8'sd0 : act_in;
              ACT_RELU6: act_out = (act_in < 8'sd0) ? 8'sd0 :
                                   (act_in > 8'(RELU6_MAX)) ? 8'(RELU6_MAX) : act_in;
              default  : act_out = act_in;  // ACT_NONE
            endcase
            spad_wr_en   = 1'b1;
            spad_wr_addr = pipe_dst_addr;
            spad_wr_data = act_out;
          end
          next_state = ST_ACT_LOOP;
        end
      end

      // ---------------------------------------------------------------
      ST_NOP_ST: begin
        busy       = 1'b1;
        next_state = ST_RETIRE;       // one idle cycle, then retire
      end

      // ---------------------------------------------------------------
      // RETIRE: commit next_pc to the program counter, return to FETCH
      ST_RETIRE: begin
        busy        = 1'b1;
        pc_write    = 1'b1;
        pc_next     = next_pc_latch;
        next_state  = ST_FETCH;
      end

      // ---------------------------------------------------------------
      // END: no RETIRE (PC does not advance); go directly to DONE
      ST_END_ST: begin
        busy       = 1'b1;
        next_state = ST_DONE_ST;
      end

      // ---------------------------------------------------------------
      // DONE: one-cycle pulse, then back to IDLE
      ST_DONE_ST: begin
        busy       = 1'b0;
        done       = 1'b1;
        next_state = ST_IDLE;
      end

      // ---------------------------------------------------------------
      // ERROR: hold until reset
      ST_ERROR_ST: begin
        error = 1'b1;
        if (fetch_error)
          error_code = ERR_FETCH_ILLEGAL;
        else if (latched_instr.decode_error)
          error_code = ERR_DECODE;
        else if (dense_error)
          error_code = ERR_DENSE_EXEC;
        else
          error_code = ERR_BAD_OPCODE;
        next_state = ST_ERROR_ST;   // hold until reset
      end

      default: next_state = ST_IDLE;
    endcase
  end

endmodule
