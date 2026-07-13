"""
TinyML Accelerator Compiler / Assembler  (Version 1 ISA)
==========================================================

Purpose
-------
Turns a description of an MLP (list of dense layers + activations) into a
stream of fixed-width instruction words that can be loaded straight into the
accelerator's Instruction Memory. This is the reference model the RTL
decoder must match bit-for-bit, so the encoding here IS the spec for the
Instruction Fetch / Instruction Decoder modules.

Instruction word format
------------------------
Every instruction starts with one 32-bit HEADER word:

    bit31..24   opcode        (8 bits)
    bit23..16   flags         (8 bits)
    bit15..8    activation    (8 bits)   (only meaningful for DENSE / ACT)
    bit7..0     reserved      (8 bits)   (must be 0)

The header is followed by zero or more 32-bit OPERAND words, depending on
opcode. Word count per opcode is fixed and known at decode time purely from
the opcode field, which is what lets the Controller FSM know how many more
words to pull from Instruction Memory before it can start executing
(FETCH -> FETCH more operands -> DECODE -> ...).

    Opcode   Mnemonic   Total words   Operand word layout
    ------   --------   -----------   --------------------------------------
    0x00     NOP        1             (none)
    0x01     LOAD       3             W1: [31:16]=mem_addr   [15:0]=sp_addr
                                       W2: [31:16]=length     [15:0]=reserved
    0x02     STORE      3             W1: [31:16]=sp_addr    [15:0]=mem_addr
                                       W2: [31:16]=length     [15:0]=reserved
    0x03     DENSE      4             W1: [31:16]=input_addr [15:0]=weight_addr
                                       W2: [31:16]=output_addr[15:0]=bias_addr
                                       W3: [31:16]=input_len  [15:0]=output_len
    0x04     ACT        2             W1: [31:16]=addr        [15:0]=length
    0x05     END        1             (none)

(Word count = 1 header word + N operand words. The decoder reads the header
first, looks up N from the opcode, then pulls N more words before decode is
complete -- this directly maps to the FETCH -> DECODE FSM states.)

All addresses / lengths are unsigned 16-bit (0..65535), matching the address
width implied by R0-R5 in the register file (section 11 of the spec).

Activation encoding (also section 9 of the spec):
    0 = NONE, 1 = RELU, 2 = RELU6, 3 = SIGMOID (future), 4 = TANH (future)

Design decision on ACT vs fused activation
-------------------------------------------
The DENSE instruction already carries an `activation` field, so the
compiler fuses activation into DENSE by default (no separate ACT
instruction needed for the common case: matmul -> bias -> activate in one
shot). The standalone ACT opcode is still fully implemented and available
for cases where an activation needs to run on data that didn't just come
out of a DENSE (e.g. re-activating a stored tensor, or future non-dense
layer types).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json

# --------------------------------------------------------------------------
# ISA constants
# --------------------------------------------------------------------------

OPCODES = {
    "NOP":   0x00,
    "LOAD":  0x01,
    "STORE": 0x02,
    "DENSE": 0x03,
    "ACT":   0x04,
    "END":   0x05,
}

WORD_COUNT = {
    "NOP": 1, "LOAD": 3, "STORE": 3, "DENSE": 4, "ACT": 2, "END": 1,
}

ACTIVATIONS = {
    "NONE": 0, "RELU": 1, "RELU6": 2, "SIGMOID": 3, "TANH": 4,
}

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def _pack16(hi: int, lo: int) -> int:
    """Pack two 16-bit fields into one 32-bit word: [31:16]=hi [15:0]=lo."""
    assert 0 <= hi <= MASK16, f"field {hi} does not fit in 16 bits"
    assert 0 <= lo <= MASK16, f"field {lo} does not fit in 16 bits"
    return ((hi & MASK16) << 16) | (lo & MASK16)


# --------------------------------------------------------------------------
# Instruction representation
# --------------------------------------------------------------------------

@dataclass
class Instruction:
    mnemonic: str
    flags: int = 0
    activation: str = "NONE"
    operands: dict = field(default_factory=dict)
    comment: str = ""

    def to_words(self) -> List[int]:
        opcode = OPCODES[self.mnemonic]
        header = ((opcode & 0xFF) << 24) | ((self.flags & 0xFF) << 16) \
                  | ((ACTIVATIONS[self.activation] & 0xFF) << 8)
        words = [header]

        if self.mnemonic == "LOAD":
            words.append(_pack16(self.operands["mem_addr"], self.operands["sp_addr"]))
            words.append(_pack16(self.operands["length"], 0))

        elif self.mnemonic == "STORE":
            words.append(_pack16(self.operands["sp_addr"], self.operands["mem_addr"]))
            words.append(_pack16(self.operands["length"], 0))

        elif self.mnemonic == "DENSE":
            words.append(_pack16(self.operands["input_addr"], self.operands["weight_addr"]))
            words.append(_pack16(self.operands["output_addr"], self.operands["bias_addr"]))
            words.append(_pack16(self.operands["input_len"], self.operands["output_len"]))

        elif self.mnemonic == "ACT":
            words.append(_pack16(self.operands["addr"], self.operands["length"]))

        # NOP / END: header only

        assert len(words) == WORD_COUNT[self.mnemonic]
        return words

    def disassemble(self) -> str:
        ops = ", ".join(f"{k}={v}" for k, v in self.operands.items())
        act = f" act={self.activation}" if self.mnemonic in ("DENSE", "ACT") else ""
        flg = f" flags=0x{self.flags:02x}" if self.flags else ""
        base = f"{self.mnemonic:6s} {ops}{act}{flg}"
        return base + (f"    ; {self.comment}" if self.comment else "")


# --------------------------------------------------------------------------
# Assembler: builds an instruction stream, emits hex / listing output
# --------------------------------------------------------------------------

class Assembler:
    def __init__(self):
        self.instructions: List[Instruction] = []

    # -- instruction emitters ------------------------------------------
    def nop(self, comment=""):
        self.instructions.append(Instruction("NOP", comment=comment))

    def load(self, mem_addr: int, sp_addr: int, length: int, comment=""):
        self.instructions.append(Instruction(
            "LOAD", operands={"mem_addr": mem_addr, "sp_addr": sp_addr, "length": length},
            comment=comment))

    def store(self, sp_addr: int, mem_addr: int, length: int, comment=""):
        self.instructions.append(Instruction(
            "STORE", operands={"sp_addr": sp_addr, "mem_addr": mem_addr, "length": length},
            comment=comment))

    def dense(self, input_addr: int, weight_addr: int, output_addr: int, bias_addr: int,
              input_len: int, output_len: int, activation: str = "NONE", flags: int = 0,
              comment: str = ""):
        self.instructions.append(Instruction(
            "DENSE", flags=flags, activation=activation,
            operands={
                "input_addr": input_addr, "weight_addr": weight_addr,
                "output_addr": output_addr, "bias_addr": bias_addr,
                "input_len": input_len, "output_len": output_len,
            }, comment=comment))

    def act(self, addr: int, length: int, activation: str, comment=""):
        self.instructions.append(Instruction(
            "ACT", activation=activation, operands={"addr": addr, "length": length},
            comment=comment))

    def end(self, comment=""):
        self.instructions.append(Instruction("END", comment=comment))

    # -- output -----------------------------------------------------------
    def to_words(self) -> List[int]:
        words = []
        for instr in self.instructions:
            words.extend(instr.to_words())
        return words

    def to_hex_lines(self) -> List[str]:
        """One 32-bit hex word per line -- directly $readmemh-compatible."""
        return [f"{w:08x}" for w in self.to_words()]

    def write_hex(self, path: str):
        with open(path, "w") as f:
            f.write("\n".join(self.to_hex_lines()) + "\n")

    def listing(self) -> str:
        lines = []
        addr = 0
        for instr in self.instructions:
            words = instr.to_words()
            lines.append(f"{addr:04d}: {instr.disassemble()}")
            addr += len(words)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Quantization helper (float32 -> INT8, symmetric per-tensor)
# --------------------------------------------------------------------------

def quantize_int8(values: List[float]):
    """Symmetric per-tensor quantization to signed INT8.

    Returns (int8_values, scale) where float_value ~= int8_value * scale.
    """
    max_abs = max(abs(v) for v in values) or 1.0
    scale = max_abs / 127.0
    q = [max(-128, min(127, round(v / scale))) for v in values]
    return q, scale


def to_hex_bytes(values: List[int]) -> List[str]:
    """Format signed 8-bit ints as two's-complement hex bytes for memory files."""
    return [f"{(v & 0xFF):02x}" for v in values]


# --------------------------------------------------------------------------
# High level: compile a whole MLP description into a program + memory map
# --------------------------------------------------------------------------

def compile_mlp(layers: List[dict], input_base: int = 0, weight_base: int = 0,
                 bias_base: int = 0, scratchpad_base: int = 0,
                 host_input_mem_addr: int = 0, host_output_mem_addr: int = 0):
    """
    layers: list of dicts, each:
        {
          "input_len": int,
          "output_len": int,
          "activation": "NONE" | "RELU" | "RELU6",
          "weights": [float, ...]   (length input_len*output_len, row-major)  [optional]
          "bias":    [float, ...]   (length output_len)                       [optional]
        }

    Returns (Assembler, memory_map dict). memory_map records where every
    layer's input/weight/bias/output live so the host driver knows where to
    DMA the quantized weights before issuing START.
    """
    asm = Assembler()
    memory_map = {"layers": []}

    # Ping-pong scratchpad buffers so layer N's output can be layer N+1's
    # input without the two ever aliasing.
    buf_a = scratchpad_base
    buf_b = scratchpad_base + max(l["input_len"] for l in layers) * 4  # generous stride

    weight_addr = weight_base
    bias_addr = bias_base

    # Step 1: bring the host's input tensor into scratchpad buffer A.
    asm.load(mem_addr=host_input_mem_addr, sp_addr=buf_a, length=layers[0]["input_len"],
             comment="load network input into scratchpad")

    cur_input_buf = buf_a
    next_output_buf = buf_b

    for i, layer in enumerate(layers):
        in_len = layer["input_len"]
        out_len = layer["output_len"]
        activation = layer.get("activation", "NONE")
        is_last = (i == len(layers) - 1)

        layer_entry = {
            "layer": i,
            "input_addr": cur_input_buf,
            "weight_addr": weight_addr,
            "bias_addr": bias_addr,
            "output_addr": next_output_buf,
            "input_len": in_len,
            "output_len": out_len,
            "activation": activation,
        }

        # Quantize + record weights/bias if provided, so the host driver can
        # write them to Weight Memory before program start.
        if "weights" in layer:
            q, scale = quantize_int8(layer["weights"])
            layer_entry["weight_hex"] = to_hex_bytes(q)
            layer_entry["weight_scale"] = scale
        if "bias" in layer:
            q, scale = quantize_int8(layer["bias"])
            layer_entry["bias_hex"] = to_hex_bytes(q)
            layer_entry["bias_scale"] = scale

        memory_map["layers"].append(layer_entry)

        asm.dense(
            input_addr=cur_input_buf, weight_addr=weight_addr,
            output_addr=next_output_buf, bias_addr=bias_addr,
            input_len=in_len, output_len=out_len, activation=activation,
            comment=f"layer {i}: dense {in_len}->{out_len} act={activation}",
        )

        # advance weight/bias allocation
        weight_addr += in_len * out_len
        bias_addr += out_len

        if is_last:
            asm.store(sp_addr=next_output_buf, mem_addr=host_output_mem_addr,
                      length=out_len, comment="store network output for host")
        else:
            # swap buffers for next layer
            cur_input_buf, next_output_buf = next_output_buf, cur_input_buf

    asm.end(comment="program complete")
    return asm, memory_map


# --------------------------------------------------------------------------
# Example: compile a 4 -> 8 -> 3 MLP  (ReLU hidden, no activation on output)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)

    layers = [
        {
            "input_len": 4, "output_len": 8, "activation": "RELU",
            "weights": [random.uniform(-1, 1) for _ in range(4 * 8)],
            "bias": [random.uniform(-1, 1) for _ in range(8)],
        },
        {
            "input_len": 8, "output_len": 3, "activation": "NONE",
            "weights": [random.uniform(-1, 1) for _ in range(8 * 3)],
            "bias": [random.uniform(-1, 1) for _ in range(3)],
        },
    ]

    asm, mem_map = compile_mlp(layers, host_input_mem_addr=0x0000, host_output_mem_addr=0x1000)

    print("=== Disassembly ===")
    print(asm.listing())
    print()
    print("=== Instruction words (hex) ===")
    print("\n".join(asm.to_hex_lines()))
    print()
    print(f"Total instruction words: {len(asm.to_words())}")
    print()
    print("=== Memory map (weights truncated) ===")
    for l in mem_map["layers"]:
        summary = {k: v for k, v in l.items() if k not in ("weight_hex", "bias_hex")}
        print(summary)
