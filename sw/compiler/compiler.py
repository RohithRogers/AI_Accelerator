"""
TinyML Accelerator Compiler / Assembler  (Version 1.1 ISA)
==========================================================

Purpose
-------
Turns a description of an MLP/CNN (dense layers, conv layers, activations)
into a stream of fixed-width instruction words that can be loaded straight
into the accelerator's Instruction Memory. This is the reference model the
RTL decoder must match bit-for-bit, so the encoding here IS the spec for
the Instruction Fetch / Instruction Decoder modules.

CHANGELOG vs V1
----------------
1. DENSE grew from 4 words to 5 words: it now carries a 32-bit M0
   (Q31 fixed-point requant multiplier) as a 5th operand word, and the
   header's "reserved" byte is repurposed as a signed shift amount `n`.
   Runtime hardware computes:  result = round((acc * M0) >> (31 + n))
   This resolves the "Open ISA item" from the V1 design doc (section 8).

2. Two new opcodes were added to support convolution without host-side
   im2col round-trips between conv layers:
       - CONV_CFG (0x06): loads per-layer conv geometry (channels,
         spatial dims, kernel size, stride, pad) into config registers.
       - CONV (0x07): triggers one convolution using the most recently
         loaded CONV_CFG geometry. Mirrors DENSE's word layout (input/
         weight/output/bias addresses + M0), with an on-chip address
         generator producing im2col addresses on the fly -- no reshaped
         copy of the feature map is ever materialized in memory.

Instruction word format
------------------------
Every instruction starts with one 32-bit HEADER word:

    bit31..24   opcode        (8 bits)
    bit23..16   flags         (8 bits)
    bit15..8    activation    (8 bits)   (meaningful for DENSE / ACT / CONV)
    bit7..0     shift (n)     (8 bits, signed two's complement)
                              (meaningful for DENSE / CONV only, else 0)

The header is followed by zero or more 32-bit OPERAND words, depending on
opcode. Word count per opcode is fixed and known at decode time purely from
the opcode field.

    Opcode   Mnemonic   Total words   Operand word layout
    ------   --------   -----------   --------------------------------------
    0x00     NOP        1             (none)
    0x01     LOAD       3             W1: [31:16]=mem_addr    [15:0]=sp_addr
                                       W2: [31:16]=length       [15:0]=reserved
    0x02     STORE      3             W1: [31:16]=sp_addr     [15:0]=mem_addr
                                       W2: [31:16]=length       [15:0]=reserved
    0x03     DENSE      5             W1: [31:16]=input_addr  [15:0]=weight_addr
                                       W2: [31:16]=output_addr [15:0]=bias_addr
                                       W3: [31:16]=input_len   [15:0]=output_len
                                       W4: M0 (32-bit, Q31 requant multiplier)
    0x04     ACT        2             W1: [31:16]=addr         [15:0]=length
    0x05     END        1             (none)
    0x06     CONV_CFG   5             W1: [31:16]=in_channels [15:0]=out_channels
                                       W2: [31:16]=h_in        [15:0]=w_in
                                       W3: [31:16]=kh          [15:0]=kw
                                       W4: [31:16]=stride      [15:0]=pad
    0x07     CONV       5             W1: [31:16]=input_addr  [15:0]=weight_addr
                                       W2: [31:16]=output_addr [15:0]=bias_addr
                                       W3: [31:16]=out_h        [15:0]=out_w
                                       W4: M0 (32-bit, Q31 requant multiplier)

All addresses / lengths / geometry fields are unsigned 16-bit (0..65535).

Activation encoding (section 9 of the design doc):
    0 = NONE, 1 = RELU, 2 = RELU6, 3 = SIGMOID (LUT), 4 = TANH (LUT)

Design decisions carried over from V1
--------------------------------------
- DENSE/CONV fuse activation + requantization into the same instruction
  (matmul/conv -> bias -> activate -> requantize, one shot). Standalone ACT
  remains available for re-activating data that didn't just come out of a
  DENSE/CONV.
- CONV_CFG is a separate instruction from CONV (rather than folding all
  geometry into one fat CONV word) so per-instruction decode cost stays
  flat regardless of kernel size, and CONV itself stays the same width as
  DENSE. CONV_CFG is issued once per conv layer; CONV can in principle be
  re-issued against the same config (not used yet, but keeps the door open
  for tiled/multi-call convolutions later).
- Padding is handled by the compiler materializing a zero border around
  each feature map in scratchpad (see `conv_scratchpad_layout`), not by
  runtime bounds-checking hardware -- consistent with how DENSE already
  handles the non-multiple-of-W tail chunk by compile-time zero-padding.
- Requantization multiplier is decomposed at compile time (Python, once)
  into (M0, n) so the runtime datapath only ever does one integer multiply
  and a rounding right-shift -- no on-chip floating point, ever.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math
import json

# --------------------------------------------------------------------------
# ISA constants
# --------------------------------------------------------------------------

OPCODES = {
    "NOP":      0x00,
    "LOAD":     0x01,
    "STORE":    0x02,
    "DENSE":    0x03,
    "ACT":      0x04,
    "END":      0x05,
    "CONV_CFG": 0x06,
    "CONV":     0x07,
}

WORD_COUNT = {
    "NOP": 1, "LOAD": 3, "STORE": 3, "DENSE": 5, "ACT": 2, "END": 1,
    "CONV_CFG": 5, "CONV": 5,
}

ACTIVATIONS = {
    "NONE": 0, "RELU": 1, "RELU6": 2, "SIGMOID": 3, "TANH": 4,
}

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# Opcodes whose header "shift" byte carries a signed requant shift `n`.
REQUANT_OPCODES = {"DENSE", "CONV"}


def _pack16(hi: int, lo: int) -> int:
    """Pack two 16-bit fields into one 32-bit word: [31:16]=hi [15:0]=lo."""
    assert 0 <= hi <= MASK16, f"field {hi} does not fit in 16 bits"
    assert 0 <= lo <= MASK16, f"field {lo} does not fit in 16 bits"
    return ((hi & MASK16) << 16) | (lo & MASK16)


def _signed8_to_field(n: int) -> int:
    """Encode a signed shift amount into an 8-bit two's-complement field."""
    assert -128 <= n <= 127, f"shift {n} does not fit in a signed 8-bit field"
    return n & 0xFF


# --------------------------------------------------------------------------
# Requantization: real_multiplier -> (M0, n) at compile time
# --------------------------------------------------------------------------

def quantize_multiplier(real_multiplier: float):
    """Decompose a positive real multiplier into (M0, n) such that

        real_multiplier ~= M0 * 2^-(31 + n)

    with M0 a Q31 fixed-point integer (M0 / 2^31 in [0.5, 1)), matching the
    TFLite/gemmlowp convention referenced in the design doc (section 8).

    Runtime hardware then computes, per output element:
        result = round((acc * M0) >> (31 + n))
        result = clamp(result, -128, 127)

    Both the multiply and the shift amount are fixed per-tensor (one M0/n
    pair per DENSE or CONV call), computed once here in Python -- the
    on-chip requantizer never does floating-point math.
    """
    if real_multiplier == 0:
        return 0, 0
    assert real_multiplier > 0, "requant multiplier must be positive"

    frac, exp = math.frexp(real_multiplier)   # real_multiplier = frac * 2**exp, 0.5<=frac<1
    m0 = round(frac * (1 << 31))
    if m0 == (1 << 31):          # rounding pushed us to exactly 1.0 in Q31
        m0 //= 2
        exp += 1
    n = -exp

    if not (-128 <= n <= 127):
        raise ValueError(
            f"requant shift n={n} out of range for real_multiplier={real_multiplier}; "
            "rescale weights/activations so the requant multiplier is closer to 1.0"
        )
    assert 0 <= m0 < (1 << 31)
    return m0, n


def requant_params(input_scale: float, weight_scale: float, output_scale: float):
    """Convenience wrapper: computes the (M0, n) pair for a DENSE/CONV layer
    directly from tensor scales, per requant_scale = (input_scale *
    weight_scale) / output_scale (design doc section 8)."""
    real_multiplier = (input_scale * weight_scale) / output_scale
    return quantize_multiplier(real_multiplier)


# --------------------------------------------------------------------------
# Instruction representation
# --------------------------------------------------------------------------

@dataclass
class Instruction:
    mnemonic: str
    flags: int = 0
    activation: str = "NONE"
    shift: int = 0                # signed n, only meaningful for DENSE/CONV
    operands: dict = field(default_factory=dict)
    comment: str = ""

    def to_words(self) -> List[int]:
        opcode = OPCODES[self.mnemonic]
        shift_field = _signed8_to_field(self.shift) if self.mnemonic in REQUANT_OPCODES else 0
        header = ((opcode & 0xFF) << 24) | ((self.flags & 0xFF) << 16) \
                  | ((ACTIVATIONS[self.activation] & 0xFF) << 8) \
                  | (shift_field & 0xFF)
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
            words.append(self.operands["m0"] & MASK32)

        elif self.mnemonic == "ACT":
            words.append(_pack16(self.operands["addr"], self.operands["length"]))

        elif self.mnemonic == "CONV_CFG":
            words.append(_pack16(self.operands["in_channels"], self.operands["out_channels"]))
            words.append(_pack16(self.operands["h_in"], self.operands["w_in"]))
            words.append(_pack16(self.operands["kh"], self.operands["kw"]))
            words.append(_pack16(self.operands["stride"], self.operands["pad"]))

        elif self.mnemonic == "CONV":
            words.append(_pack16(self.operands["input_addr"], self.operands["weight_addr"]))
            words.append(_pack16(self.operands["output_addr"], self.operands["bias_addr"]))
            words.append(_pack16(self.operands["out_h"], self.operands["out_w"]))
            words.append(self.operands["m0"] & MASK32)

        # NOP / END: header only

        assert len(words) == WORD_COUNT[self.mnemonic], \
            f"{self.mnemonic}: expected {WORD_COUNT[self.mnemonic]} words, got {len(words)}"
        return words

    def disassemble(self) -> str:
        ops = ", ".join(f"{k}={v}" for k, v in self.operands.items())
        act = f" act={self.activation}" if self.mnemonic in ("DENSE", "ACT", "CONV") else ""
        shf = f" shift={self.shift}" if self.mnemonic in REQUANT_OPCODES else ""
        flg = f" flags=0x{self.flags:02x}" if self.flags else ""
        base = f"{self.mnemonic:9s} {ops}{act}{shf}{flg}"
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
              input_scale: float = 1.0, weight_scale: float = 1.0, output_scale: float = 1.0,
              comment: str = ""):
        """requant multiplier = (input_scale * weight_scale) / output_scale,
        decomposed at compile time into (M0, shift) -- see quantize_multiplier()."""
        m0, n = requant_params(input_scale, weight_scale, output_scale)
        self.instructions.append(Instruction(
            "DENSE", flags=flags, activation=activation, shift=n,
            operands={
                "input_addr": input_addr, "weight_addr": weight_addr,
                "output_addr": output_addr, "bias_addr": bias_addr,
                "input_len": input_len, "output_len": output_len,
                "m0": m0,
            }, comment=comment))

    def act(self, addr: int, length: int, activation: str, comment=""):
        self.instructions.append(Instruction(
            "ACT", activation=activation, operands={"addr": addr, "length": length},
            comment=comment))

    def conv_cfg(self, in_channels: int, out_channels: int, h_in: int, w_in: int,
                 kh: int, kw: int, stride: int, pad: int, comment: str = ""):
        self.instructions.append(Instruction(
            "CONV_CFG",
            operands={
                "in_channels": in_channels, "out_channels": out_channels,
                "h_in": h_in, "w_in": w_in, "kh": kh, "kw": kw,
                "stride": stride, "pad": pad,
            }, comment=comment))

    def conv(self, input_addr: int, weight_addr: int, output_addr: int, bias_addr: int,
              out_h: int, out_w: int, activation: str = "NONE", flags: int = 0,
              input_scale: float = 1.0, weight_scale: float = 1.0, output_scale: float = 1.0,
              comment: str = ""):
        """Must be preceded by a matching CONV_CFG in the same instruction
        stream. out_h/out_w are the *output* feature map spatial dims
        (compiler-computed from CONV_CFG's geometry + stride/pad), not
        re-derived on-chip, so the address generator never needs a divider."""
        m0, n = requant_params(input_scale, weight_scale, output_scale)
        self.instructions.append(Instruction(
            "CONV", flags=flags, activation=activation, shift=n,
            operands={
                "input_addr": input_addr, "weight_addr": weight_addr,
                "output_addr": output_addr, "bias_addr": bias_addr,
                "out_h": out_h, "out_w": out_w, "m0": m0,
            }, comment=comment))

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
# Conv scratchpad layout: bake the zero border into the physical layout so
# the on-chip address generator never needs runtime bounds-checking.
# --------------------------------------------------------------------------

def conv_scratchpad_layout(channels: int, h: int, w: int, pad: int):
    """Returns (padded_h, padded_w, per_channel_words) for a feature map
    stored with a physical zero border of `pad` on every side. The host/
    compiler is responsible for zero-filling the border once and writing
    real data into the interior -- the CONV address generator then always
    reads in-bounds addresses, exactly mirroring how DENSE's compiler
    zero-pads its final input chunk instead of adding runtime masking."""
    padded_h = h + 2 * pad
    padded_w = w + 2 * pad
    return padded_h, padded_w, channels * padded_h * padded_w


def conv_output_dims(h_in: int, w_in: int, kh: int, kw: int, stride: int, pad: int):
    """Standard conv output size formula -- computed here at compile time so
    the instruction stream carries out_h/out_w explicitly and the hardware
    address generator never needs a divider."""
    out_h = (h_in + 2 * pad - kh) // stride + 1
    out_w = (w_in + 2 * pad - kw) // stride + 1
    return out_h, out_w


# --------------------------------------------------------------------------
# High level: compile a whole MLP description into a program + memory map
# --------------------------------------------------------------------------

def compile_mlp(layers: List[dict], input_base: int = 0, weight_base: int = 0,
                 bias_base: int = 0, scratchpad_base: int = 0,
                 host_input_mem_addr: int = 0, host_output_mem_addr: int = 0,
                 input_scale: float = 1.0):
    """
    layers: list of dicts, each:
        {
          "input_len": int,
          "output_len": int,
          "activation": "NONE" | "RELU" | "RELU6",
          "weights": [float, ...]   (length input_len*output_len, row-major)
          "bias":    [float, ...]   (length output_len)
          "output_scale": float     (scale of this layer's INT8 output;
                                      required so the requant multiplier
                                      can be computed -- see section 8)
        }

    input_scale: scale of the INT8 tensor the host LOADs in as network
    input. Each subsequent layer's input_scale is the previous layer's
    output_scale, threaded automatically.

    Returns (Assembler, memory_map dict). memory_map records where every
    layer's input/weight/bias/output live, plus each layer's (M0, shift)
    requant params, so the host driver knows where to DMA the quantized
    weights before issuing START.
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
    cur_input_scale = input_scale

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

        weight_scale = 1.0
        if "weights" in layer:
            q, weight_scale = quantize_int8(layer["weights"])
            layer_entry["weight_hex"] = to_hex_bytes(q)
            layer_entry["weight_scale"] = weight_scale
        if "bias" in layer:
            q, bias_scale = quantize_int8(layer["bias"])
            layer_entry["bias_hex"] = to_hex_bytes(q)
            layer_entry["bias_scale"] = bias_scale

        output_scale = layer.get("output_scale", 1.0)
        m0, n = requant_params(cur_input_scale, weight_scale, output_scale)
        layer_entry["input_scale"] = cur_input_scale
        layer_entry["output_scale"] = output_scale
        layer_entry["requant_m0"] = m0
        layer_entry["requant_shift"] = n

        memory_map["layers"].append(layer_entry)

        asm.dense(
            input_addr=cur_input_buf, weight_addr=weight_addr,
            output_addr=next_output_buf, bias_addr=bias_addr,
            input_len=in_len, output_len=out_len, activation=activation,
            input_scale=cur_input_scale, weight_scale=weight_scale, output_scale=output_scale,
            comment=f"layer {i}: dense {in_len}->{out_len} act={activation}",
        )

        # advance weight/bias allocation
        weight_addr += in_len * out_len
        bias_addr += out_len
        cur_input_scale = output_scale

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
# plus a standalone CONV_CFG/CONV example to exercise the new opcodes.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)

    layers = [
        {
            "input_len": 4, "output_len": 8, "activation": "RELU",
            "weights": [random.uniform(-1, 1) for _ in range(4 * 8)],
            "bias": [random.uniform(-1, 1) for _ in range(8)],
            "output_scale": 0.05,
        },
        {
            "input_len": 8, "output_len": 3, "activation": "NONE",
            "weights": [random.uniform(-1, 1) for _ in range(8 * 3)],
            "bias": [random.uniform(-1, 1) for _ in range(3)],
            "output_scale": 0.03,
        },
    ]

    asm, mem_map = compile_mlp(layers, host_input_mem_addr=0x0000, host_output_mem_addr=0x1000,
                                input_scale=0.02)

    print("=== MLP disassembly ===")
    print(asm.listing())
    print()
    print(f"Total instruction words: {len(asm.to_words())}")
    print()
    print("=== Memory map (weights truncated) ===")
    for l in mem_map["layers"]:
        summary = {k: v for k, v in l.items() if k not in ("weight_hex", "bias_hex")}
        print(summary)

    # ---- CONV / CONV_CFG smoke test -------------------------------------
    # 3-channel 8x8 input -> 3x3 conv, 8 output channels, stride 1, pad 1
    print()
    print("=== CONV example: CONV_CFG + CONV (in isolation) ===")
    conv_asm = Assembler()
    IC, OC, H, W, KH, KW, STRIDE, PAD = 3, 8, 8, 8, 3, 3, 1, 1
    out_h, out_w = conv_output_dims(H, W, KH, KW, STRIDE, PAD)
    padded_h, padded_w, feat_words = conv_scratchpad_layout(IC, H, W, PAD)

    conv_asm.conv_cfg(in_channels=IC, out_channels=OC, h_in=H, w_in=W,
                       kh=KH, kw=KW, stride=STRIDE, pad=PAD,
                       comment="configure conv1 geometry")
    conv_asm.conv(input_addr=0x0000, weight_addr=0x2000, output_addr=0x4000, bias_addr=0x3000,
                  out_h=out_h, out_w=out_w, activation="RELU",
                  input_scale=0.02, weight_scale=0.01, output_scale=0.04,
                  comment="conv1: 3x8x8 -> 8x{}x{}".format(out_h, out_w))
    conv_asm.end(comment="conv program complete")
    print(conv_asm.listing())
    print(f"padded feature map: {IC}x{padded_h}x{padded_w} ({feat_words} INT8 words incl. zero border)")
    print(f"output dims: {OC}x{out_h}x{out_w}")
