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
        elif mnemonic == "ACT":
            fields["addr"], fields["length"] = unpack16(operand_words[0])

        decoded.append({
            "mnemonic": mnemonic, "flags": flags,
            "activation": ACT_NAMES[act] if mnemonic in ("DENSE", "ACT") else None,
            **fields,
        })
    return decoded


def main():
    layers = [
        {"input_len": 4, "output_len": 8, "activation": "RELU"},
        {"input_len": 8, "output_len": 3, "activation": "NONE"},
    ]
    asm, _ = compile_mlp(layers, host_input_mem_addr=0x0000, host_output_mem_addr=0x1000)

    words = asm.to_words()
    decoded = decode_stream(words)

    print("=== Decoded instruction stream ===")
    for d in decoded:
        print(d)

    # Cross-check: number of decoded instructions matches what we assembled
    assert len(decoded) == len(asm.instructions), "instruction count mismatch after decode"

    for original, redecoded in zip(asm.instructions, decoded):
        assert original.mnemonic == redecoded["mnemonic"], "opcode mismatch"
        for key, value in original.operands.items():
            assert redecoded[key] == value, f"field {key} mismatch: {value} != {redecoded[key]}"

    print("\nRound-trip check passed: encode -> words -> decode matches original instructions.")


if __name__ == "__main__":
    main()
