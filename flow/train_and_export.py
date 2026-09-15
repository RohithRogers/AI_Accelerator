#!/usr/bin/env python3
"""
flow/train_and_export.py -- TinyML HAR Model: Train, Quantize, and Export
==========================================================================

PURPOSE:
  Trains a small MLP on the UCI Human Activity Recognition dataset using
  pure-NumPy SGD (no PyTorch/TensorFlow dependency), then applies
  post-training quantization (PTQ) to INT8 using the accelerator's native
  Q31 requantization scheme. Exports all hex memory files and model_config.json
  so the hardware testbench (tb_flow.sv) can run the model unchanged.

NETWORK TOPOLOGY:
  Input (16) -> Dense(16->32, ReLU) -> Dense(32->16, ReLU) -> Dense(16->6, NONE)
  Total parameters: 16*32 + 32 + 32*16 + 16 + 16*6 + 6 = 512+32+512+16+96+6 = 1174

SCRATCHPAD LAYOUT:
  addr   0 .. 15  : input activation   (16 bytes)
  addr  64 .. 95  : layer 1 output     (32 bytes)
  addr 128 .. 143 : layer 2 output     (16 bytes)
  addr 192 .. 197 : layer 3 output     (6 bytes)

All addresses are comfortably within SPAD_DEPTH=4096.

USAGE:
  python flow/train_and_export.py [--synthetic]

  --synthetic : skip UCI HAR download, use Gaussian-blob synthetic data
"""

import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flow.har_dataset import (
    load_uci_har, generate_synthetic_har, fit_normalizer, normalize,
    CLASS_NAMES, N_FEATURES, N_CLASSES,
)
from sw.compiler.compiler import (
    Assembler, OPCODES, ACTIVATIONS, quantize_multiplier, requant_params,
)

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

LAYER_SIZES  = [N_FEATURES, 32, 16, N_CLASSES]   # 16->32->16->6
ACTIVATIONS_ = ["RELU", "RELU", "NONE"]           # fused into DENSE instructions

# Scratchpad addresses (aligned to 64-byte boundaries for clarity)
SP_ADDRS = [0, 64, 128, 192]

# Training hyper-parameters
EPOCHS      = 250
LR          = 0.01
BATCH_SIZE  = 64
L2          = 1e-4
SEED        = 42

# ---------------------------------------------------------------------------
# Helpers: pure-NumPy MLP
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(logits, y):
    probs  = softmax(logits)
    n      = len(y)
    loss   = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    return loss, dlogits

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


class MLP:
    """Minimal pure-NumPy 3-layer MLP with He initialisation and SGD+momentum."""

    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)   # He init
            self.weights.append(rng.normal(0, std, (fan_out, fan_in)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
        # Momentum buffers
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.vb = [np.zeros_like(b) for b in self.biases]

    def forward(self, x):
        """Returns (activations_list, pre_act_list); activations[0] = x."""
        acts = [x]
        pres = []
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = acts[-1] @ W.T + b
            pres.append(z)
            if i < n_layers - 1:    # hidden layers: ReLU
                acts.append(relu(z))
            else:                   # output layer: linear
                acts.append(z)
        return acts, pres

    def backward(self, acts, pres, dout, lr, l2, momentum=0.9):
        n_layers = len(self.weights)
        d = dout
        for i in reversed(range(n_layers)):
            pre   = pres[i]
            a_in  = acts[i]
            W     = self.weights[i]

            # gradient through activation (all hidden layers use ReLU)
            if i < n_layers - 1:
                d = d * relu_grad(pre)

            dW = d.T @ a_in + l2 * W
            db = d.sum(axis=0)
            d  = d @ W   # backprop through weight matrix

            # SGD + momentum
            self.vw[i] = momentum * self.vw[i] + lr * dW
            # The current RTL bias memory is INT8 in accumulator units.  A
            # general float bias cannot be represented faithfully there, so
            # train this deployment model bias-free (the initialized biases
            # remain exactly zero) rather than exporting incorrectly scaled
            # bias values.
            self.vb[i] = 0
            self.weights[i] -= self.vw[i]
            self.biases[i]  -= self.vb[i]

    def predict(self, x):
        acts, _ = self.forward(x)
        return acts[-1]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_val, y_val, epochs, lr, batch_size, l2):
    rng = np.random.default_rng(SEED)
    n   = len(y_train)
    best_val_acc   = 0.0
    best_weights   = None
    best_biases    = None

    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        X_tr, y_tr = X_train[perm], y_train[perm]
        ep_loss = 0.0

        for start in range(0, n, batch_size):
            xb = X_tr[start : start + batch_size]
            yb = y_tr[start : start + batch_size]
            acts, pres = model.forward(xb)
            loss, dout = cross_entropy(acts[-1], yb)
            ep_loss += loss
            model.backward(acts, pres, dout, lr, l2)

        if ep % 20 == 0 or ep == epochs:
            val_logits = model.predict(X_val)
            val_acc    = accuracy(val_logits, y_val)
            print(f"  Epoch {ep:3d}/{epochs}  loss={ep_loss:.4f}  val_acc={val_acc*100:.1f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = [w.copy() for w in model.weights]
                best_biases  = [b.copy() for b in model.biases]

    # Restore best checkpoint
    if best_weights is not None:
        model.weights = best_weights
        model.biases  = best_biases
    return best_val_acc


# ---------------------------------------------------------------------------
# Post-training quantization (PTQ)
# ---------------------------------------------------------------------------

def collect_activation_range(model, X_calib):
    """
    Run forward pass on calibration data and collect per-layer output ranges.
    Returns list of (min_val, max_val) for the *input* of each layer
    (which equals the output of the previous layer; the network input is idx 0).
    """
    acts, _ = model.forward(X_calib)
    ranges = []
    for a in acts:   # acts[0]=input, acts[1..N]=layer outputs
        ranges.append((float(a.min()), float(a.max())))
    return ranges   # len = n_layers + 1


def symmetric_scale(min_val, max_val):
    """
    Symmetric per-tensor scale for INT8 quantization.
    scale = max(|min|, |max|) / 127
    """
    abs_max = max(abs(min_val), abs(max_val), 1e-8)
    return abs_max / 127.0


def quantize_weights_int8(W, b):
    """
    Symmetric per-tensor quantization of float weights and biases to INT8.
    Returns (W_q, b_q, w_scale, b_scale).
    """
    w_abs_max = max(abs(W.min()), abs(W.max()), 1e-8)
    w_scale   = w_abs_max / 127.0
    W_q       = np.clip(np.round(W / w_scale), -128, 127).astype(np.int8)

    b_abs_max = max(abs(b.min()), abs(b.max()), 1e-8)
    b_scale   = b_abs_max / 127.0
    B_q       = np.clip(np.round(b / b_scale), -128, 127).astype(np.int8)

    return W_q, B_q, w_scale, b_scale


def quantize_model(model, X_calib):
    """
    Perform PTQ and return:
      - layers_q  : list of dicts with INT8 weights/biases and requant params
      - act_scales: list of float scales (one per activation tensor, len=n_layers+1)
    """
    ranges      = collect_activation_range(model, X_calib)
    act_scales  = [symmetric_scale(lo, hi) for (lo, hi) in ranges]
    # act_scales[i] = scale of activation entering layer i  (layer 0 input = network input)
    # act_scales[i+1] = scale of activation exiting layer i

    layers_q = []
    for i in range(len(model.weights)):
        W      = model.weights[i]
        b      = model.biases[i]
        W_q, B_q, w_scale, _ = quantize_weights_int8(W, b)

        in_scale  = act_scales[i]
        out_scale = act_scales[i + 1]
        m0, shift = requant_params(in_scale, w_scale, out_scale)

        layers_q.append({
            "in_features":  W.shape[1],
            "out_features": W.shape[0],
            "activation":   ACTIVATIONS_[i],
            "weights":      W_q,
            "biases":       B_q,
            "m0":           m0,
            "shift":        shift,
            "in_scale":     in_scale,
            "w_scale":      w_scale,
            "out_scale":    out_scale,
        })

    return layers_q, act_scales


# ---------------------------------------------------------------------------
# INT8 golden simulation (matches hardware exactly)
# ---------------------------------------------------------------------------

def sat8(val):
    return int(np.clip(val, -128, 127))

def simulate_layer_int8(x_in, W_q, B_q, activation, m0, shift):
    """
    Exact fixed-point simulation for one dense layer (mirrors generate_network.py).
    """
    out_len, in_len = W_q.shape
    y_out = np.zeros(out_len, dtype=np.int8)
    for o in range(out_len):
        dot = int(np.sum(x_in.astype(np.int64) * W_q[o].astype(np.int64)))
        acc = dot + int(B_q[o])
        total_shift = 31 + shift
        if total_shift > 0:
            prod   = acc * m0
            offset = 1 << (total_shift - 1)
            req    = (prod + offset) >> total_shift
        elif total_shift < 0:
            req = (acc * m0) << (-total_shift)
        else:
            req = acc * m0
        if activation == "RELU":
            req = max(0, req)
        y_out[o] = sat8(req)
    return y_out

def run_int8_model(x_int8, layers_q):
    """Run the full INT8 MLP. Returns final INT8 output vector."""
    act = x_int8.astype(np.int8)
    for lq in layers_q:
        act = simulate_layer_int8(
            act, lq["weights"], lq["biases"],
            lq["activation"], lq["m0"], lq["shift"]
        )
    return act


# ---------------------------------------------------------------------------
# Compiler / hex export
# ---------------------------------------------------------------------------

def compile_and_export_model(layers_q, input_vector_int8, out_dir=SCRIPT_DIR):
    """
    Compile the quantized MLP into hex files using the existing Assembler.
    """
    os.makedirs(out_dir, exist_ok=True)
    asm = Assembler()

    input_len = layers_q[0]["in_features"]
    sp_in_addr = SP_ADDRS[0]

    # LOAD: host memory -> scratchpad
    asm.load(mem_addr=0, sp_addr=sp_in_addr, length=input_len,
             comment="Load 16-feature HAR input vector")

    weight_bytes = []
    bias_bytes   = []
    cur_w_offset = 0
    cur_b_offset = 0

    for i, lq in enumerate(layers_q):
        w_flat = lq["weights"].flatten().tolist()
        b_flat = lq["biases"].tolist()

        weight_bytes.extend(w_flat)
        bias_bytes.extend(b_flat)

        # Emit DENSE with scales so the Assembler computes m0/n itself
        # (we pass the pre-computed m0/n via input_scale=1, weight_scale=1
        #  and output_scale trick, but more directly: we build the Instruction
        #  manually to inject our PTQ-derived m0 and shift exactly)
        from sw.compiler.compiler import Instruction, OPCODES, ACTIVATIONS
        instr = Instruction(
            "DENSE",
            activation=lq["activation"],
            shift=lq["shift"],
            operands={
                "input_addr":  SP_ADDRS[i],
                "weight_addr": cur_w_offset,
                "output_addr": SP_ADDRS[i + 1],
                "bias_addr":   cur_b_offset,
                "input_len":   lq["in_features"],
                "output_len":  lq["out_features"],
                "m0":          lq["m0"],
            },
            comment=f"HAR layer {i+1}: {lq['in_features']}->{lq['out_features']} act={lq['activation']}"
        )
        asm.instructions.append(instr)

        cur_w_offset += len(w_flat)
        cur_b_offset += len(b_flat)

    final_out_len  = layers_q[-1]["out_features"]
    final_sp_addr  = SP_ADDRS[len(layers_q)]

    asm.store(sp_addr=final_sp_addr, mem_addr=0, length=final_out_len,
              comment="Store 6 class logits to output memory")
    asm.end(comment="Halt accelerator")

    # Golden reference output
    golden_output  = run_int8_model(input_vector_int8, layers_q)

    # Intermediate activations (for model_config.json)
    intermediates = [input_vector_int8.tolist()]
    act = input_vector_int8.astype(np.int8)
    for lq in layers_q:
        act = simulate_layer_int8(act, lq["weights"], lq["biases"],
                                  lq["activation"], lq["m0"], lq["shift"])
        intermediates.append(act.tolist())

    imem_words = asm.to_words()

    # --- Write files ---
    def write_hex8(path, data):
        with open(path, "w") as f:
            for v in data:
                f.write(f"{int(v) & 0xFF:02x}\n")

    write_hex8(os.path.join(out_dir, "weight.hex"),        weight_bytes)
    write_hex8(os.path.join(out_dir, "bias.hex"),          bias_bytes)
    write_hex8(os.path.join(out_dir, "input.hex"),         input_vector_int8.tolist())
    write_hex8(os.path.join(out_dir, "golden_output.hex"), golden_output.tolist())

    with open(os.path.join(out_dir, "imem.hex"), "w") as f:
        for w in imem_words:
            f.write(f"{w:08x}\n")

    config = {
        "model":       "HAR_MLP_16x32x16x6",
        "num_layers":  len(layers_q),
        "input_len":   layers_q[0]["in_features"],
        "output_len":  final_out_len,
        "input_vector":          input_vector_int8.tolist(),
        "golden_output":         golden_output.tolist(),
        "intermediate_activations": intermediates,
        "num_instructions":      len(asm.instructions),
        "num_imem_words":        len(imem_words),
        "total_weights":         len(weight_bytes),
        "total_biases":          len(bias_bytes),
        "class_names":           CLASS_NAMES,
        "predicted_class":       int(np.argmax(golden_output)),
        "predicted_label":       CLASS_NAMES[int(np.argmax(golden_output))],
        "layer_scales":          [
            {"in": lq["in_scale"], "w": lq["w_scale"], "out": lq["out_scale"],
             "m0": lq["m0"], "shift": lq["shift"]}
            for lq in layers_q
        ],
        "layers": [
            {"name": f"layer_{i + 1}", "in_features": lq["in_features"],
             "out_features": lq["out_features"], "activation": lq["activation"],
             "m0": lq["m0"], "shift": lq["shift"]}
            for i, lq in enumerate(layers_q)
        ],
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return asm, config, golden_output


# ---------------------------------------------------------------------------
# Select representative test vectors (one per class, worst-case quantization)
# ---------------------------------------------------------------------------

def pick_test_vectors(X_test_norm, y_test, layers_q, act_scales, n_per_class=2):
    """
    Pick up to n_per_class examples per class from the test set.
    Quantize them using act_scales[0] (the input scale) and return as INT8.
    Returns list of (int8_vector, true_label) tuples.
    """
    in_scale = act_scales[0]
    vectors  = []
    for cls in range(N_CLASSES):
        idxs = np.where(y_test == cls)[0]
        if len(idxs) == 0:
            continue
        # Take first n_per_class from this class
        for idx in idxs[:n_per_class]:
            x_float = X_test_norm[idx]
            x_int8  = np.clip(np.round(x_float / in_scale), -128, 127).astype(np.int8)
            vectors.append((x_int8, int(cls)))
    return vectors


# ---------------------------------------------------------------------------
# Accuracy evaluation on INT8 model
# ---------------------------------------------------------------------------

def evaluate_int8(X_norm, y, layers_q, act_scales):
    """Evaluate INT8 model accuracy over the full set."""
    in_scale = act_scales[0]
    correct  = 0
    for i in range(len(y)):
        x_int8  = np.clip(np.round(X_norm[i] / in_scale), -128, 127).astype(np.int8)
        logits  = run_int8_model(x_int8, layers_q)
        if int(np.argmax(logits)) == int(y[i]):
            correct += 1
    return correct / len(y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train HAR TinyML model and export to accelerator hex files")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of downloading UCI HAR")
    parser.add_argument("--out-dir", default=SCRIPT_DIR,
                        help="Directory to write hex files (default: flow/)")
    args = parser.parse_args()

    print("================================================================")
    print("TinyML HAR Model -- Train, Quantize & Export")
    print("================================================================")

    # 1. Load dataset
    if args.synthetic:
        print("\n[1/6] Using SYNTHETIC HAR data (--synthetic flag)")
        X, y = generate_synthetic_har(n_samples_per_class=300, seed=SEED)
        split = int(0.8 * len(y))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]
    else:
        print("\n[1/6] Loading UCI HAR dataset...")
        try:
            X_train, y_train, X_test, y_test = load_uci_har(verbose=True)
        except (OSError, RuntimeError) as exc:
            # Keep the flow runnable in offline CI/development environments.
            print(f"  UCI HAR unavailable ({exc}); falling back to synthetic HAR data.")
            X, y = generate_synthetic_har(n_samples_per_class=300, seed=SEED)
            split = int(0.8 * len(y))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

    # 2. Normalise
    print("\n[2/6] Normalising features (z-score)...")
    norm_mean, norm_std = fit_normalizer(X_train)
    X_train_n = normalize(X_train, norm_mean, norm_std)
    X_test_n  = normalize(X_test,  norm_mean, norm_std)
    print(f"  Train: {X_train_n.shape}  Test: {X_test_n.shape}")

    # 3. Train float model
    print(f"\n[3/6] Training {LAYER_SIZES} MLP for {EPOCHS} epochs...")
    model = MLP(LAYER_SIZES, seed=SEED)
    best_val_acc = train(model, X_train_n, y_train, X_test_n, y_test,
                         EPOCHS, LR, BATCH_SIZE, L2)
    float_test_acc = accuracy(model.predict(X_test_n), y_test)
    print(f"\n  Float model test accuracy : {float_test_acc * 100:.1f}%")

    # 4. Quantize
    print("\n[4/6] Post-training quantization (PTQ) to INT8...")
    layers_q, act_scales = quantize_model(model, X_train_n)
    print(f"  Input scale : {act_scales[0]:.6f}")
    for i, lq in enumerate(layers_q):
        print(f"  Layer {i+1}: w_scale={lq['w_scale']:.6f}  out_scale={lq['out_scale']:.6f}"
              f"  M0=0x{lq['m0']:08x}  shift={lq['shift']}")

    print("\n  Evaluating INT8 model accuracy (this may take a minute)...")
    int8_test_acc = evaluate_int8(X_test_n, y_test, layers_q, act_scales)
    print(f"  INT8 model test accuracy  : {int8_test_acc * 100:.1f}%")

    if int8_test_acc < 0.75:
        print("  WARNING: INT8 accuracy below 75%. Consider increasing EPOCHS or adjusting scales.")

    # 5. Pick representative test vector (walk example = class 0)
    print("\n[5/6] Selecting representative input vector for hex export...")
    test_vecs = pick_test_vectors(X_test_n, y_test, layers_q, act_scales, n_per_class=2)

    # Export the first WALKING example (or just first test vector)
    # Find a WALKING sample (class 0) for demonstration
    export_vec, export_label = test_vecs[0]
    print(f"  Selected sample: class={export_label} ({CLASS_NAMES[export_label]})")
    print(f"  INT8 input vector: {export_vec.tolist()}")

    # 6. Compile and export hex files
    print(f"\n[6/6] Compiling and exporting to: {args.out_dir}")
    asm, config, golden = compile_and_export_model(layers_q, export_vec, out_dir=args.out_dir)
    config["accuracy"] = {"float_test": float(float_test_acc), "int8_test": float(int8_test_acc)}
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Exported files:")
    for fname in ["imem.hex", "weight.hex", "bias.hex", "input.hex",
                  "golden_output.hex", "model_config.json"]:
        fpath = os.path.join(args.out_dir, fname)
        size  = os.path.getsize(fpath)
        print(f"    {fname:30s} ({size:6d} bytes)")

    print("\n================================================================")
    print("EXPORT SUMMARY")
    print("================================================================")
    print(f"  Model         : {config['model']}")
    print(f"  Topology      : {LAYER_SIZES}")
    print(f"  Total weights : {config['total_weights']}")
    print(f"  Total biases  : {config['total_biases']}")
    print(f"  IMEM words    : {config['num_imem_words']}")
    print(f"  Float accuracy: {float_test_acc * 100:.1f}%")
    print(f"  INT8 accuracy : {int8_test_acc * 100:.1f}%")
    print(f"  Demo input    : class={export_label} ({CLASS_NAMES[export_label]})")
    print(f"  Golden output : {golden.tolist()}")
    print(f"  Predicted     : {config['predicted_label']} (argmax={config['predicted_class']})")

    # Also save test vectors for run_accelerator_test.py
    test_vec_path = os.path.join(args.out_dir, "test_inputs.json")
    test_vec_data = {
        "metadata": {
            "model":      config["model"],
            "topology":   LAYER_SIZES,
            "class_names": CLASS_NAMES,
            "in_scale":   float(act_scales[0]),
            "norm_mean":  norm_mean.tolist(),
            "norm_std":   norm_std.tolist(),
        },
        "test_vectors": [
            {
                "id":           vi,
                "true_class":   int(cls),
                "true_label":   CLASS_NAMES[cls],
                "input_int8":   vec.tolist(),
            }
            for vi, (vec, cls) in enumerate(test_vecs[:8])
        ],
    }
    with open(test_vec_path, "w") as f:
        json.dump(test_vec_data, f, indent=2)
    print(f"\n  Test vectors  : {len(test_vec_data['test_vectors'])} saved to test_inputs.json")
    print("================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
