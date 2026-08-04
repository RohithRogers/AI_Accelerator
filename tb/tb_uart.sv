// =============================================================================
// tb_uart.sv — Testbench for UART Host Interface & Memory Bus Integration
// =============================================================================

`timescale 1ns/1ps
import tinyml_pkg::*;

module tb_uart;

  localparam int CLK_FREQ   = 50_000_000;
  localparam int BAUD_RATE  = 115200;
  localparam int BIT_PERIOD = 1_000_000_000 / BAUD_RATE; // ~8680 ns

  logic clk;
  logic rst_n;
  logic rx;
  logic tx;

  logic       start_pulse;
  logic       busy, done, error;
  logic [7:0] error_code;

  // Memory Controller Bus
  logic        host_wr_en, host_rd_en;
  logic [2:0]  host_target_id;
  logic [15:0] host_addr;
  logic [31:0] host_wr_data32, host_rd_data32;
  logic signed [7:0] host_wr_data8, host_rd_data8;

  // Memory interconnect ports
  logic        imem_wr_en, imem_rd_en;
  logic [9:0]  imem_wr_addr, imem_rd_addr;
  logic [31:0] imem_wr_data, imem_rd_data;

  logic        hmem_b_rd_en, hmem_b_wr_en;
  logic [11:0] hmem_b_rd_addr, hmem_b_wr_addr;
  logic signed [7:0] hmem_b_rd_data, hmem_b_wr_data;

  logic        spad_b_rd_en, spad_b_wr_en;
  logic [11:0] spad_b_rd_addr, spad_b_wr_addr;
  logic signed [7:0] spad_b_rd_data, spad_b_wr_data;

  logic        weight_rd_en, weight_wr_en;
  logic [15:0] weight_rd_addr, weight_wr_addr;
  logic signed [7:0] weight_rd_data, weight_wr_data;

  logic        bias_rd_en, bias_wr_en;
  logic [15:0] bias_rd_addr, bias_wr_addr;
  logic signed [7:0] bias_rd_data, bias_wr_data;

  logic        outmem_rd_en;
  logic [11:0] outmem_rd_addr;
  logic signed [7:0] outmem_rd_data;

  // Start pulse latch for test verification
  logic start_seen;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_seen <= 1'b0;
    end else if (start_pulse) begin
      start_seen <= 1'b1;
    end
  end

  // UART Interface Module
  uart_interface #(
    .CLK_FREQ(CLK_FREQ),
    .BAUD_RATE(BAUD_RATE)
  ) u_uart_if (
    .clk(clk), .rst_n(rst_n),
    .rx(rx), .tx(tx),
    .start_pulse(start_pulse),
    .busy(busy), .done(done), .error(error), .error_code(error_code),
    .host_wr_en(host_wr_en), .host_rd_en(host_rd_en),
    .host_target_id(host_target_id), .host_addr(host_addr),
    .host_wr_data32(host_wr_data32), .host_wr_data8(host_wr_data8),
    .host_rd_data32(host_rd_data32), .host_rd_data8(host_rd_data8)
  );

  // Memory controller module
  memory_controller u_mem_ctrl (
    .clk(clk), .rst_n(rst_n), .busy(busy),
    .host_wr_en(host_wr_en), .host_rd_en(host_rd_en),
    .host_target_id(host_target_id), .host_addr(host_addr),
    .host_wr_data32(host_wr_data32), .host_wr_data8(host_wr_data8),
    .host_rd_data32(host_rd_data32), .host_rd_data8(host_rd_data8),

    .imem_host_wr_en(imem_wr_en), .imem_host_wr_addr(imem_wr_addr), .imem_host_wr_data(imem_wr_data),
    .imem_host_rd_en(imem_rd_en), .imem_host_rd_addr(imem_rd_addr), .imem_host_rd_data(imem_rd_data),

    .hmem_host_wr_en(hmem_b_wr_en), .hmem_host_wr_addr(hmem_b_wr_addr), .hmem_host_wr_data(hmem_b_wr_data),
    .hmem_host_rd_en(hmem_b_rd_en), .hmem_host_rd_addr(hmem_b_rd_addr), .hmem_host_rd_data(hmem_b_rd_data),

    .spad_host_wr_en(spad_b_wr_en), .spad_host_wr_addr(spad_b_wr_addr), .spad_host_wr_data(spad_b_wr_data),
    .spad_host_rd_en(spad_b_rd_en), .spad_host_rd_addr(spad_b_rd_addr), .spad_host_rd_data(spad_b_rd_data),

    .weight_host_wr_en(weight_wr_en), .weight_host_wr_addr(weight_wr_addr), .weight_host_wr_data(weight_wr_data),
    .weight_host_rd_en(weight_rd_en), .weight_host_rd_addr(weight_rd_addr), .weight_host_rd_data(weight_rd_data),

    .bias_host_wr_en(bias_wr_en), .bias_host_wr_addr(bias_wr_addr), .bias_host_wr_data(bias_wr_data),
    .bias_host_rd_en(bias_rd_en), .bias_host_rd_addr(bias_rd_addr), .bias_host_rd_data(bias_rd_data),

    .outmem_host_rd_en(outmem_rd_en), .outmem_host_rd_addr(outmem_rd_addr), .outmem_host_rd_data(outmem_rd_data)
  );

  // Memories
  instruction_memory #(.DEPTH(1024)) u_imem (
    .clk(clk), .rst_n(rst_n),
    .rd_en(imem_rd_en), .rd_addr(imem_rd_addr), .rd_data(imem_rd_data),
    .wr_en(imem_wr_en), .wr_addr(imem_wr_addr), .wr_data(imem_wr_data)
  );

  weight_memory #(.DEPTH(65536)) u_weight (
    .clk(clk), .rst_n(rst_n),
    .rd_en(weight_rd_en), .rd_addr(weight_rd_addr), .rd_data(weight_rd_data),
    .wr_en(weight_wr_en), .wr_addr(weight_wr_addr), .wr_data(weight_wr_data)
  );

  always #10 clk = ~clk; // 50 MHz clock

  // UART send byte task
  task send_uart_byte(input [7:0] data);
    integer i;
    begin
      rx = 1'b0; // Start bit
      #(BIT_PERIOD);
      for (i = 0; i < 8; i = i + 1) begin
        rx = data[i];
        #(BIT_PERIOD);
      end
      rx = 1'b1; // Stop bit
      #(BIT_PERIOD);
    end
  endtask

  // UART receive byte task
  task recv_uart_byte(output [7:0] data);
    integer i;
    begin
      @(negedge tx); // Wait for start bit
      #(BIT_PERIOD / 2); // Center of start bit
      #(BIT_PERIOD); // Center of bit 0
      for (i = 0; i < 8; i = i + 1) begin
        data[i] = tx;
        #(BIT_PERIOD);
      end
      #(BIT_PERIOD / 2); // Center of stop bit
    end
  endtask

  logic [7:0] rxb;

  initial begin
    clk = 0; rst_n = 0; rx = 1; busy = 0; done = 0; error = 0; error_code = 0;
    #100 rst_n = 1;
    #200;

    $display("=== TEST 1: Send UART WRITE_MEM packet for Weight Memory (Target 3) ===");
    // CMD=0x01, target=0x03, addr=0x0010, len=0x0002, data=[0xA5, 0x5A]
    send_uart_byte(8'h01);
    send_uart_byte(8'h03);
    send_uart_byte(8'h00);
    send_uart_byte(8'h10);
    send_uart_byte(8'h00);
    send_uart_byte(8'h02);
    send_uart_byte(8'hA5);
    send_uart_byte(8'h5A);

    #(BIT_PERIOD * 5); // Allow UART state machine to settle back to IDLE

    $display("=== TEST 2: Send UART READ_MEM packet & receive bytes over TX ===");
    // CMD=0x02, target=0x03, addr=0x0010, len=0x0002
    fork
      begin
        send_uart_byte(8'h02);
        send_uart_byte(8'h03);
        send_uart_byte(8'h00);
        send_uart_byte(8'h10);
        send_uart_byte(8'h00);
        send_uart_byte(8'h02);
      end
      begin
        recv_uart_byte(rxb);
        assert(rxb == 8'hA5) else $error("UART read byte 0 mismatch: 0x%02h", rxb);
        recv_uart_byte(rxb);
        assert(rxb == 8'h5A) else $error("UART read byte 1 mismatch: 0x%02h", rxb);
      end
    join

    #(BIT_PERIOD * 5);

    $display("=== TEST 3: Send UART START Command (0x03) ===");
    send_uart_byte(8'h03);
    #(BIT_PERIOD * 2);
    assert(start_seen == 1'b1) else $error("start_pulse failed to trigger");

    $display("=== ALL UART INTERFACE TESTS PASSED ===");
    $finish;
  end

endmodule
