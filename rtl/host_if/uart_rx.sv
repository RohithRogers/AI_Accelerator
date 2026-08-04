// =============================================================================
// uart_rx.sv — UART 8N1 Serial Receiver
//
// PURPOSE:
//   Samples incoming serial line `rx` at `BAUD_RATE` and outputs received bytes
//   with a one-cycle `rx_ready` pulse.
// =============================================================================

`timescale 1ns/1ps

module uart_rx #(
  parameter int unsigned CLK_FREQ  = 50_000_000,
  parameter int unsigned BAUD_RATE = 115200
) (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       rx,

  output logic [7:0] rx_data,
  output logic       rx_ready
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
  logic [7:0] rx_shift;
  logic       rx_sync0, rx_sync1; // 2-stage synchronizer for meta-stability

  // Synchronize rx input
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_sync0 <= 1'b1;
      rx_sync1 <= 1'b1;
    end else begin
      rx_sync0 <= rx;
      rx_sync1 <= rx_sync0;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= STATE_IDLE;
      clk_cnt  <= '0;
      bit_idx  <= '0;
      rx_shift <= '0;
      rx_data  <= '0;
      rx_ready <= 1'b0;
    end else begin
      rx_ready <= 1'b0; // Default pulse low

      case (state)
        STATE_IDLE: begin
          clk_cnt <= '0;
          bit_idx <= '0;
          if (rx_sync1 == 1'b0) begin // Start bit falling edge
            state <= STATE_START_BIT;
          end
        end

        STATE_START_BIT: begin
          if (clk_cnt == (CLKS_PER_BIT - 1) / 2) begin
            if (rx_sync1 == 1'b0) begin // Validate mid-start bit
              clk_cnt <= '0;
              state   <= STATE_DATA_BITS;
            end else begin
              state <= STATE_IDLE; // False start
            end
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        STATE_DATA_BITS: begin
          if (clk_cnt == CLKS_PER_BIT - 1) begin
            clk_cnt <= '0;
            rx_shift[bit_idx] <= rx_sync1;
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
          if (clk_cnt == CLKS_PER_BIT - 1) begin
            state    <= STATE_IDLE;
            clk_cnt  <= '0;
            rx_data  <= rx_shift;
            rx_ready <= 1'b1; // Pulse data ready
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        default: state <= STATE_IDLE;
      endcase
    end
  end

endmodule
