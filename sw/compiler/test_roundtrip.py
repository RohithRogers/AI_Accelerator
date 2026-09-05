"""
Round-trip check: decode the emitted instruction words back into fields and
confirm they match what was encoded. This is the same decode logic the RTL
Instruction Decoder must implement, expressed in Python so both sides can be
checked against one source of truth.
"""

from sw.compiler.compiler import compile_mlp, OPCODES, WORD_COUNT, ACTIVATIONS

OPCODE_NAMES = {v: k for k, v in OPCODES.items()}
ACT_NAMES = {v: k for k, v in ACTIVATIONS.items()}


def unpack16(word):
    return (word >> 16) & 0xFFFF, word & 0xFFFF


def decode_stream(words):
    """Walk a flat instruction-word stream and yield decoded instructions."""
    i = 0
    decoded = []
    while i < len(words):
        header = words[i]
        opcode = (header >> 24) & 0xFF
        flags = (header >> 16) & 0xFF
        act = (header >> 8) & 0xFF
        
        # In V1.1, the lower 8 bits of header is a signed shift amount `n`
        shift = header & 0xFF
        if shift & 0x80:
            shift = shift - 256

        mnemonic = OPCODE_NAMES[opcode]
        n = WORD_COUNT[mnemonic]
        operand_words = words[i + 1:i + n]
        i += n

        fields = {}
        if mnemonic == "LOAD":
            fields["mem_addr"], fields["sp_addr"] = unpack16(operand_words[0])
            fields["length"], _ = unpack16(operand_words[1])
        elif mnemonic == "STORE":
            fields["sp_addr"], fields["mem_addr"] = unpack16(operand_words[0])
            fields["length"], _ = unpack16(operand_words[1])
        elif mnemonic == "DENSE":
            fields["input_addr"], fields["weight_addr"] = unpack16(operand_words[0])
            fields["output_addr"], fields["bias_addr"] = unpack16(operand_words[1])
            fields["input_len"], fields["output_len"] = unpack16(operand_words[2])
            fields["m0"] = operand_words[3]
        elif mnemonic == "ACT":
            fields["addr"], fields["length"] = unpack16(operand_words[0])
        elif mnemonic == "CONV_CFG":
            fields["in_channels"], fields["out_channels"] = unpack16(operand_words[0])
            fields["h_in"], fields["w_in"] = unpack16(operand_words[1])
            fields["kh"], fields["kw"] = unpack16(operand_words[2])
            fields["stride"], fields["pad"] = unpack16(operand_words[3])
        elif mnemonic == "CONV":
            fields["input_addr"], fields["weight_addr"] = unpack16(operand_words[0])
            fields["output_addr"], fields["bias_addr"] = unpack16(operand_words[1])
            fields["out_h"], fields["out_w"] = unpack16(operand_words[2])
            fields["m0"] = operand_words[3]

        decoded.append({
            "mnemonic": mnemonic, "flags": flags,
            "activation": ACT_NAMES[act] if mnemonic in ("DENSE", "ACT", "CONV") else "NONE",
            "shift": shift if mnemonic in ("DENSE", "CONV") else 0,
            **fields,
        })
    return decoded


def main():
    layers = [
        {"input_len": 4, "output_len": 8, "activation": "RELU"},
        {"input_len": 8, "output_len": 3, "activation": "NONE"},
    ]
    asm, _ = compile_mlp(layers, host_input_mem_addr=0x0000, host_output_mem_addr=0x1000)

    # Add CONV_CFG and CONV smoke test instructions to test roundtrip for new opcodes
    asm.conv_cfg(in_channels=3, out_channels=8, h_in=8, w_in=8, kh=3, kw=3, stride=1, pad=1)
    asm.conv(input_addr=0x0000, weight_addr=0x2000, output_addr=0x4000, bias_addr=0x3000, out_h=8, out_w=8, activation="RELU")

    words = asm.to_words()
    decoded = decode_stream(words)

    print("=== Decoded instruction stream ===")
    for d in decoded:
        print(d)

    # Cross-check: number of decoded instructions matches what we assembled
    assert len(decoded) == len(asm.instructions), "instruction count mismatch after decode"

    for original, redecoded in zip(asm.instructions, decoded):
        assert original.mnemonic == redecoded["mnemonic"], "opcode mismatch"
        assert redecoded["activation"] == original.activation, f"activation mismatch: {original.activation} != {redecoded['activation']}"
        assert redecoded["shift"] == original.shift, f"shift mismatch: {original.shift} != {redecoded['shift']}"
        assert redecoded["flags"] == original.flags, f"flags mismatch: {original.flags} != {redecoded['flags']}"
        for key, value in original.operands.items():
            assert redecoded[key] == value, f"field {key} mismatch: {value} != {redecoded[key]}"

    print("\nRound-trip check passed: encode -> words -> decode matches original instructions.")



if __name__ == "__main__":
    main()
