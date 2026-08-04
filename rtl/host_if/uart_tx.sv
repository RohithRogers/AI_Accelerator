// =============================================================================
// uart_tx.sv — UART 8N1 Serial Transmitter
//
// PURPOSE:
//   Transmits 8-bit bytes serially over `tx` line at `BAUD_RATE`.
//   `tx_busy` is high while transmitting.
// =============================================================================

`timescale 1ns/1ps

module uart_tx #(
  parameter int unsigned CLK_FREQ  = 50_000_000,
  parameter int unsigned BAUD_RATE = 115200
) (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       tx_start,
  input  logic [7:0] tx_data,

  output logic       tx,
  output logic       tx_busy
);

  localparam int CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

  typedef enum logic [1:0] {
    STATE_IDLE      = 2'b00,
    STATE_START_BIT = 2'b01,
    STATE_DATA_BITS = 2'b10,
    STATE_STOP_BIT  = 2'b11
  } state_t;

  state_t state;
  logic [$clog2(CLKS_PER_BIT)-1:0] clk_cnt;
  logic [2:0] bit_idx;
  logic [7:0] tx_shift;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= STATE_IDLE;
      clk_cnt  <= '0;
      bit_idx  <= '0;
      tx_shift <= '0;
      tx       <= 1'b1; // Idle state for UART tx is HIGH
      tx_busy  <= 1'b0;
    end else begin
      case (state)
        STATE_IDLE: begin
          tx      <= 1'b1;
          tx_busy <= 1'b0;
          clk_cnt <= '0;
          bit_idx <= '0;
          if (tx_start) begin
            tx_shift <= tx_data;
            tx_busy  <= 1'b1;
            state    <= STATE_START_BIT;
          end
        end

        STATE_START_BIT: begin
          tx <= 1'b0; // Start bit is LOW
          if (clk_cnt == CLKS_PER_BIT - 1) begin
            clk_cnt <= '0;
            state   <= STATE_DATA_BITS;
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        STATE_DATA_BITS: begin
          tx <= tx_shift[bit_idx];
          if (clk_cnt == CLKS_PER_BIT - 1) begin
            clk_cnt <= '0;
            if (bit_idx == 3'd7) begin
              bit_idx <= '0;
              state   <= STATE_STOP_BIT;
            end else begin
              bit_idx <= bit_idx + 1'b1;
            end
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        STATE_STOP_BIT: begin
          tx <= 1'b1; // Stop bit is HIGH
          if (clk_cnt == CLKS_PER_BIT - 1) begin
            state   <= STATE_IDLE;
            tx_busy <= 1'b0;
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        default: state <= STATE_IDLE;
      endcase
    end
  end

endmodule
