#!/usr/bin/env python3
"""
flow/generate_network.py — TinyML Accelerator Network Generator & Hex Exporter
=============================================================================

PURPOSE:
  Defines a custom multi-layer Neural Network (MLP), compiles the execution instructions,
  initializes input activations, weights, and biases, and exports ready-to-load hexadecimal
  memory files for the SystemVerilog hardware accelerator and the Python golden simulator:
    - imem.hex          : 32-bit instruction words
    - weight.hex        : 8-bit signed weight bytes
    - bias.hex          : 8-bit signed bias bytes
    - input.hex         : 8-bit signed input vector
    - golden_output.hex : 8-bit signed expected reference output bytes
    - model_config.json : JSON summary of topology and metadata
"""

import os
import sys
import json
import numpy as np

# Ensure workspace root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sw.compiler.compiler import (
    Assembler,
    OPCODES,
    ACTIVATIONS,
    quantize_multiplier,
    requant_params,
)


def sat8(val):
    """Saturate signed 32-bit integer to signed INT8 [-128, 127]."""
    return int(np.clip(val, -128, 127))


def apply_act(val, act_type):
    """Apply activation function on INT8 scalar/array."""
    if act_type == "RELU":
        return np.maximum(0, val)
    elif act_type == "RELU6":
        return np.clip(val, 0, 127)
    return val


def simulate_layer(x_in, weights, biases, act_type, m0, shift):
    """
    Exact fixed-point hardware reference model for one Dense layer:
      acc = bias + dot(x_in, weight_row)
      requant = round((acc * m0) >> (31 + shift))
      out = sat8(activate(requant))
    """
    out_len, in_len = weights.shape
    y_out = np.zeros(out_len, dtype=np.int8)

    for o in range(out_len):
        dot = np.sum(x_in.astype(np.int64) * weights[o, :].astype(np.int64))
        acc = dot + int(biases[o])
        
        # Runtime requantization (Q31 multiplication + shift)
        total_shift = 31 + shift
        if total_shift > 0:
            prod = acc * m0
            # Fixed-point rounding (add 1 << (total_shift - 1))
            offset = 1 << (total_shift - 1)
            requant = (prod + offset) >> total_shift
        elif total_shift < 0:
            requant = (acc * m0) << (-total_shift)
        else:
            requant = acc * m0
        
        # Activation and saturation
        activated = apply_act(requant, act_type)
        y_out[o] = sat8(activated)

    return y_out


def build_custom_mlp():
    """
    Define custom MLP architecture:
      Input (4) -> Layer 1 (3, ReLU) -> Layer 2 (2, None) -> Output (2)
    You can easily modify this function to build any arbitrary topology!
    """
    np.random.seed(42)

    # 1. Network Topology Definition
    layers_def = [
        {
            "name": "layer1",
            "in_features": 4,
            "out_features": 3,
            "activation": "RELU",
            "m0": 0x40000000,   # Identity scaling with shift = -1
            "shift": -1,
            # Weights shape: (out_features, in_features)
            "weights": np.array([
                [ 1,  2, -1,  1],
                [-2,  1,  1,  0],
                [ 3, -1,  0,  2]
            ], dtype=np.int8),
            "biases": np.array([5, 1, -2], dtype=np.int8),
        },
        {
            "name": "layer2",
            "in_features": 3,
            "out_features": 2,
            "activation": "NONE",
            "m0": 0x40000000,
            "shift": -1,
            "weights": np.array([
                [ 2, -1,  3],
                [-1,  4, -2]
            ], dtype=np.int8),
            "biases": np.array([-3, 5], dtype=np.int8),
        },
    ]

    # Input activation vector
    input_vector = np.array([1, -2, 3, 2], dtype=np.int8)

    return layers_def, input_vector


def compile_and_export(layers_def, input_vector, out_dir=SCRIPT_DIR):
    """Compile MLP into instructions, allocate memories, and export hex files."""
    os.makedirs(out_dir, exist_ok=True)
    asm = Assembler()

    # Memory Layout Allocation
    # Scratchpad:
    #   0..63   : Layer 0 input (input_vector)
    #   64..79  : Layer 1 output
    #   80..95  : Layer 2 output
    # Host Data Memory:
    #   0..N-1  : Preloaded input vector
    # Output Memory:
    #   0..M-1  : Final output tensor
    # Weight Memory:
    #   0..     : Concatenated weight bytes (row-major per layer)
    # Bias Memory:
    #   0..     : Concatenated bias bytes

    input_len = len(input_vector)
    sp_in_addr = 0
    sp_layer_addrs = [0, 64, 80]
    
    # 1. Emit LOAD instruction (Host RAM -> Scratchpad)
    asm.load(mem_addr=0, sp_addr=sp_in_addr, length=input_len, comment="Load input vector")

    weight_bytes = []
    bias_bytes = []
    weight_offsets = []
    bias_offsets = []

    cur_w_offset = 0
    cur_b_offset = 0

    # 2. Emit DENSE instructions for each layer
    for i, l in enumerate(layers_def):
        w = l["weights"].flatten()
        b = l["biases"]

        weight_offsets.append(cur_w_offset)
        bias_offsets.append(cur_b_offset)

        weight_bytes.extend(w.tolist())
        bias_bytes.extend(b.tolist())

        asm.dense(
            input_addr=sp_layer_addrs[i],
            weight_addr=cur_w_offset,
            output_addr=sp_layer_addrs[i + 1],
            bias_addr=cur_b_offset,
            input_len=l["in_features"],
            output_len=l["out_features"],
            activation=l["activation"],
            shift=l["shift"],
            m0=l["m0"],
            comment=f"Dense layer {i+1} ({l['name']})"
        )

        cur_w_offset += len(w)
        cur_b_offset += len(b)

    final_out_len = layers_def[-1]["out_features"]
    final_sp_addr = sp_layer_addrs[len(layers_def)]

    # 3. Emit STORE instruction (Scratchpad -> Output Memory)
    asm.store(sp_addr=final_sp_addr, mem_addr=0, length=final_out_len, comment="Store final predictions")

    # 4. Emit END instruction
    asm.end(comment="Halt accelerator")

    # --------------------------------------------------------------------------
    # Compute Golden Reference Output
    # --------------------------------------------------------------------------
    cur_act = input_vector
    intermediates = [input_vector.tolist()]
    for l in layers_def:
        cur_act = simulate_layer(
            cur_act, l["weights"], l["biases"], l["activation"], l["m0"], l["shift"]
        )
        intermediates.append(cur_act.tolist())

    golden_output = cur_act

    # --------------------------------------------------------------------------
    # Export Hex Files
    # --------------------------------------------------------------------------
    # 1. imem.hex (32-bit hex words)
    imem_words = asm.to_words()
    imem_path = os.path.join(out_dir, "imem.hex")
    with open(imem_path, "w") as f:
        for w in imem_words:
            f.write(f"{w:08x}\n")

    # 2. weight.hex (8-bit hex bytes)
    weight_path = os.path.join(out_dir, "weight.hex")
    with open(weight_path, "w") as f:
        for b in weight_bytes:
            f.write(f"{b & 0xFF:02x}\n")

    # 3. bias.hex (8-bit hex bytes)
    bias_path = os.path.join(out_dir, "bias.hex")
    with open(bias_path, "w") as f:
        for b in bias_bytes:
            f.write(f"{b & 0xFF:02x}\n")

    # 4. input.hex (8-bit hex bytes)
    input_path = os.path.join(out_dir, "input.hex")
    with open(input_path, "w") as f:
        for b in input_vector:
            f.write(f"{b & 0xFF:02x}\n")

    # 5. golden_output.hex (8-bit hex bytes)
    golden_path = os.path.join(out_dir, "golden_output.hex")
    with open(golden_path, "w") as f:
        for b in golden_output:
            f.write(f"{b & 0xFF:02x}\n")

    # 6. model_config.json metadata
    config_path = os.path.join(out_dir, "model_config.json")
    config = {
        "num_layers": len(layers_def),
        "input_len": input_len,
        "output_len": final_out_len,
        "input_vector": input_vector.tolist(),
        "golden_output": golden_output.tolist(),
        "intermediate_activations": intermediates,
        "num_instructions": len(asm.instructions),
        "num_imem_words": len(imem_words),
        "total_weights": len(weight_bytes),
        "total_biases": len(bias_bytes),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("================================================================")
    print("TinyML Accelerator — Network Generation & Export Complete")
    print("================================================================")
    print(f"  Instructions generated : {len(asm.instructions)} ({len(imem_words)} 32-bit words)")
    print(f"  Input features         : {input_vector.tolist()}")
    print(f"  Golden expected output : {golden_output.tolist()}")
    print(f"  Exported files:")
    print(f"    - {imem_path}")
    print(f"    - {weight_path}")
    print(f"    - {bias_path}")
    print(f"    - {input_path}")
    print(f"    - {golden_path}")
    print(f"    - {config_path}")
    print("================================================================")

    return asm, config


if __name__ == "__main__":
    layers, inp = build_custom_mlp()
    compile_and_export(layers, inp)

