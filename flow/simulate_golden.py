#!/usr/bin/env python3
"""
flow/simulate_golden.py — Software Golden Reference Model Simulator
===================================================================

PURPOSE:
  Loads the generated neural network memory files (input.hex, weight.hex, bias.hex,
  and model_config.json) and simulates the exact INT8 fixed-point arithmetic model
  implemented in the hardware accelerator. Prints layer-by-layer activations and
  verifies numerical correctness against golden_output.hex.
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_hex_bytes(filepath):
    """Load signed 8-bit integers from a 2-digit hex file."""
    values = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required hex file: {filepath}")
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            u8 = int(line, 16)
            s8 = u8 - 256 if u8 >= 128 else u8
            values.append(s8)
    return np.array(values, dtype=np.int8)


def load_imem_words(filepath):
    """Load 32-bit hex words from instruction memory file."""
    words = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required instruction file: {filepath}")
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            words.append(int(line, 16))
    return words


def sat8(val):
    return int(np.clip(val, -128, 127))


def apply_act(val, act_type):
    if act_type == "RELU":
        return np.maximum(0, val)
    elif act_type == "RELU6":
        return np.clip(val, 0, 127)
    return val


def simulate_dense_layer(x_in, weights, biases, act_type, m0, shift):
    """
    Exact hardware bit-level simulation for a single dense layer:
      1. Dot product: sum(x_in * weight_row) in signed 64/32-bit domain.
      2. Bias addition: acc = dot + bias.
      3. Requantization: round((acc * m0) >> (31 + shift)) with Q31 multiplier.
      4. Activation: ReLU, ReLU6, or None.
      5. INT8 Clamping: saturate to [-128, 127].
    """
    out_len, in_len = weights.shape
    y_out = np.zeros(out_len, dtype=np.int8)
    raw_accs = []
    requants = []

    for o in range(out_len):
        dot = int(np.sum(x_in.astype(np.int64) * weights[o, :].astype(np.int64)))
        acc = dot + int(biases[o])
        raw_accs.append(acc)

        total_shift = 31 + shift
        if total_shift > 0:
            prod = acc * m0
            offset = 1 << (total_shift - 1)
            requant = (prod + offset) >> total_shift
        elif total_shift < 0:
            requant = (acc * m0) << (-total_shift)
        else:
            requant = acc * m0

        requants.append(requant)
        activated = apply_act(requant, act_type)
        y_out[o] = sat8(activated)

    return y_out, raw_accs, requants


def run_golden_simulation(flow_dir=SCRIPT_DIR):
    config_file = os.path.join(flow_dir, "model_config.json")
    input_file = os.path.join(flow_dir, "input.hex")
    weight_file = os.path.join(flow_dir, "weight.hex")
    bias_file = os.path.join(flow_dir, "bias.hex")
    golden_file = os.path.join(flow_dir, "golden_output.hex")

    print("================================================================")
    print("TinyML Accelerator — Python Golden Reference Model Simulation")
    print("================================================================")

    # 1. Load configuration and memory files
    with open(config_file, "r") as f:
        config = json.load(f)

    input_vec = load_hex_bytes(input_file)
    weights_all = load_hex_bytes(weight_file)
    biases_all = load_hex_bytes(bias_file)
    golden_expected = load_hex_bytes(golden_file)

    print(f"Loaded {len(input_vec)} input bytes: {input_vec.tolist()}")
    print(f"Loaded {len(weights_all)} total weight bytes and {len(biases_all)} bias bytes.")
    print("----------------------------------------------------------------")

    # 2. Simulate layer by layer.  Trained exports are self-describing;
    # retain the dummy-model fallback for existing legacy images.
    layers_def = config.get("layers")
    if layers_def is None:
        from flow.generate_network import build_custom_mlp
        layers_def, _ = build_custom_mlp()

    cur_act = input_vec
    w_ptr = 0
    b_ptr = 0

    for i, l in enumerate(layers_def):
        in_dim = l["in_features"]
        out_dim = l["out_features"]
        act = l["activation"]
        m0 = l["m0"]
        shift = l["shift"]

        w_count = in_dim * out_dim
        b_count = out_dim

        w_layer = weights_all[w_ptr : w_ptr + w_count].reshape((out_dim, in_dim))
        b_layer = biases_all[b_ptr : b_ptr + b_count]

        w_ptr += w_count
        b_ptr += b_count

        out_act, accs, reqs = simulate_dense_layer(cur_act, w_layer, b_layer, act, m0, shift)

        print(f"Layer {i+1} [{l.get('name', f'layer_{i+1}')}]: {in_dim} -> {out_dim} (Activation: {act})")
        print(f"  Input        : {cur_act.tolist()}")
        print(f"  Raw Accum    : {accs}")
        print(f"  Requantized  : {reqs}")
        print(f"  Output INT8  : {out_act.tolist()}")
        print("----------------------------------------------------------------")

        cur_act = out_act

    # 3. Verify against golden_output.hex
    mismatch = np.where(cur_act != golden_expected)[0]
    if len(mismatch) == 0:
        print(">>> GOLDEN SIMULATION MATCHES EXPECTED REFERENCE OUTPUT! <<<")
        print(f"Final Model Prediction: {cur_act.tolist()}")
        print("================================================================")
        return True
    else:
        print(f"!!! MISMATCH DETECTED at indices {mismatch} !!!")
        print(f"  Simulated: {cur_act.tolist()}")
        print(f"  Expected : {golden_expected.tolist()}")
        print("================================================================")
        return False


if __name__ == "__main__":
    success = run_golden_simulation()
    if not success:
        sys.exit(1)

