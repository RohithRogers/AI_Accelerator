// =============================================================================
// uart_interface.sv — Top-Level Host UART Bridge & Command Decoder
//
// PURPOSE:
//   Parses incoming UART byte packets from `rx`, generates host memory access
//   commands to `memory_controller`, triggers top-level execution via `start`,
//   and streams requested readback bytes or status out over `tx`.
//
// PACKET PROTOCOL:
//   - CMD 0x01 (WRITE_MEM): [0x01] [target_id] [addr_hi] [addr_lo] [len_hi] [len_lo] [bytes...]
//   - CMD 0x02 (READ_MEM) : [0x02] [target_id] [addr_hi] [addr_lo] [len_hi] [len_lo]
//   - CMD 0x03 (START)    : [0x03]
//   - CMD 0x04 (STATUS)   : [0x04] -> Responds with [status_flags] [error_code]
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module uart_interface #(
  parameter int unsigned CLK_FREQ  = 50_000_000,
  parameter int unsigned BAUD_RATE = 115200
) (
  input  logic clk,
  input  logic rst_n,

  // Serial lines
  input  logic rx,
  output logic tx,

  // Top-level accelerator handshake
  output logic       start_pulse,
  input  logic       busy,
  input  logic       done,
  input  logic       error,
  input  logic [7:0] error_code,

  // Memory controller interface
  output logic        host_wr_en,
  output logic        host_rd_en,
  output logic [2:0]  host_target_id,
  output logic [15:0] host_addr,
  output logic [31:0] host_wr_data32,
  output logic signed [7:0] host_wr_data8,
  input  logic [31:0] host_rd_data32,
  input  logic signed [7:0] host_rd_data8
);

  // Submodules
  logic [7:0] rx_data;
  logic       rx_ready;
  logic [7:0] tx_data;
  logic       tx_start;
  logic       tx_busy;

  uart_rx #(
    .CLK_FREQ(CLK_FREQ),
    .BAUD_RATE(BAUD_RATE)
  ) u_rx (
    .clk(clk), .rst_n(rst_n),
    .rx(rx), .rx_data(rx_data), .rx_ready(rx_ready)
  );

  uart_tx #(
    .CLK_FREQ(CLK_FREQ),
    .BAUD_RATE(BAUD_RATE)
  ) u_tx (
    .clk(clk), .rst_n(rst_n),
    .tx_start(tx_start), .tx_data(tx_data),
    .tx(tx), .tx_busy(tx_busy)
  );

  // State Machine
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_GET_TARGET,
    ST_GET_ADDR_HI,
    ST_GET_ADDR_LO,
    ST_GET_LEN_HI,
    ST_GET_LEN_LO,
    ST_WRITE_BYTE,
    ST_WRITE_IMEM_ACCUM,
    ST_READ_ISSUE,
    ST_READ_WAIT,
    ST_READ_TX,
    ST_TX_STATUS_0,
    ST_TX_STATUS_1_WAIT,
    ST_TX_STATUS_1
  } state_t;

  state_t state;

  // Registers
  logic [7:0]  cmd_reg;
  logic [2:0]  target_reg;
  logic [15:0] addr_reg;
  logic [15:0] len_reg;
  logic [15:0] byte_cnt;
  logic [1:0]  imem_byte_idx;
  logic [31:0] imem_accum;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= ST_IDLE;
      start_pulse    <= 1'b0;
      host_wr_en     <= 1'b0;
      host_rd_en     <= 1'b0;
      host_target_id <= 3'd0;
      host_addr      <= 16'h0;
      host_wr_data32 <= 32'h0;
      host_wr_data8  <= 8'h0;
      tx_start       <= 1'b0;
      tx_data        <= 8'h0;
      cmd_reg        <= 8'h0;
      target_reg     <= 3'd0;
      addr_reg       <= 16'h0;
      len_reg        <= 16'h0;
      byte_cnt       <= 16'h0;
      imem_byte_idx  <= 2'd0;
      imem_accum     <= 32'h0;
    end else begin
      start_pulse <= 1'b0;
      host_wr_en  <= 1'b0;
      host_rd_en  <= 1'b0;
      tx_start    <= 1'b0;

      case (state)
        ST_IDLE: begin
          imem_byte_idx <= 2'd0;
          byte_cnt      <= 16'h0;
          if (rx_ready) begin
            cmd_reg <= rx_data;
            if (rx_data == 8'h01 || rx_data == 8'h02) begin
              state <= ST_GET_TARGET;
            end else if (rx_data == 8'h03) begin
              start_pulse <= 1'b1; // Trigger start pulse
            end else if (rx_data == 8'h04) begin
              // Read status
              if (!tx_busy) begin
                tx_data  <= {busy, done, error, 5'b00000};
                tx_start <= 1'b1;
                state    <= ST_TX_STATUS_1_WAIT;
              end
            end
          end
        end

        ST_GET_TARGET: begin
          if (rx_ready) begin
            target_reg <= rx_data[2:0];
            state      <= ST_GET_ADDR_HI;
          end
        end

        ST_GET_ADDR_HI: begin
          if (rx_ready) begin
            addr_reg[15:8] <= rx_data;
            state          <= ST_GET_ADDR_LO;
          end
        end

        ST_GET_ADDR_LO: begin
          if (rx_ready) begin
            addr_reg[7:0] <= rx_data;
            state         <= ST_GET_LEN_HI;
          end
        end

        ST_GET_LEN_HI: begin
          if (rx_ready) begin
            len_reg[15:8] <= rx_data;
            state         <= ST_GET_LEN_LO;
          end
        end

        ST_GET_LEN_LO: begin
          if (rx_ready) begin
            len_reg[7:0] <= rx_data;
            if (cmd_reg == 8'h01) begin
              state <= (target_reg == 3'd0) ? ST_WRITE_IMEM_ACCUM : ST_WRITE_BYTE;
            end else begin
              state <= ST_READ_ISSUE;
            end
          end
        end

        // Write 8-bit memory stream (HMEM/SPAD/WEIGHT/BIAS)
        ST_WRITE_BYTE: begin
          if (rx_ready) begin
            host_wr_en     <= 1'b1;
            host_target_id <= target_reg;
            host_addr      <= addr_reg;
            host_wr_data8  <= rx_data;
            addr_reg       <= addr_reg + 1'b1;
            byte_cnt       <= byte_cnt + 1'b1;
            if (byte_cnt + 1'b1 == len_reg) begin
              state <= ST_IDLE;
            end
          end
        end

        // Write 32-bit IMEM stream (accumulate 4 bytes per word, Big-Endian)
        ST_WRITE_IMEM_ACCUM: begin
          if (rx_ready) begin
            case (imem_byte_idx)
              2'd0: imem_accum[31:24] <= rx_data;
              2'd1: imem_accum[23:16] <= rx_data;
              2'd2: imem_accum[15:8]  <= rx_data;
              2'd3: begin
                imem_accum[7:0] <= rx_data;
                host_wr_en     <= 1'b1;
                host_target_id <= 3'd0;
                host_addr      <= addr_reg;
                host_wr_data32 <= {imem_accum[31:8], rx_data};
                addr_reg       <= addr_reg + 1'b1;
              end
            endcase
            imem_byte_idx <= imem_byte_idx + 1'b1;
            byte_cnt      <= byte_cnt + 1'b1;
            if (byte_cnt + 1'b1 == len_reg) begin
              state <= ST_IDLE;
            end
          end
        end

        // Read stream from memory
        ST_READ_ISSUE: begin
          if (byte_cnt < len_reg) begin
            host_rd_en     <= 1'b1;
            host_target_id <= target_reg;
            host_addr      <= addr_reg;
            state          <= ST_READ_WAIT;
          end else begin
            state <= ST_IDLE;
          end
        end

        ST_READ_WAIT: begin
          // Wait 1 cycle for registered read latency
          state <= ST_READ_TX;
        end

        ST_READ_TX: begin
          if (!tx_busy) begin
            tx_data  <= (target_reg == 3'd0) ? host_rd_data32[7:0] : host_rd_data8;
            tx_start <= 1'b1;
            addr_reg <= addr_reg + 1'b1;
            byte_cnt <= byte_cnt + 1'b1;
            state    <= ST_READ_ISSUE;
          end
        end

        ST_TX_STATUS_1_WAIT: begin
          if (!tx_busy) begin
            state <= ST_TX_STATUS_1;
          end
        end

        ST_TX_STATUS_1: begin
          if (!tx_busy) begin
            tx_data  <= error_code;
            tx_start <= 1'b1;
            state    <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
