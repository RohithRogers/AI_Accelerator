#!/usr/bin/env python3
"""Run every exported representative vector through the RTL testbench."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / "flow"

RTL_SOURCES = [
    "rtl/tinyml_pkg.sv", "rtl/frontend/program_counter.sv",
    "rtl/frontend/instruction_memory.sv", "rtl/frontend/instruction_fetch.sv",
    "rtl/frontend/instruction_decoder.sv", "rtl/frontend/controller_fsm.sv",
    "rtl/memory/scratchpad_memory.sv", "rtl/memory/weight_memory.sv",
    "rtl/memory/bias_memory.sv", "rtl/memory/host_data_memory.sv",
    "rtl/memory/output_memory.sv", "rtl/memory/memory_controller.sv",
    "rtl/compute/simd_mac_array.sv", "rtl/compute/accumulator.sv",
    "rtl/compute/bias_adder.sv", "rtl/compute/requantizer_runtime.sv",
    "rtl/compute/activation_unit.sv", "rtl/compute/writeback_unit.sv",
    "rtl/compute/dense_engine.sv", "rtl/top/tinyml_accelerator_top.sv", "flow/tb_flow.sv",
]


def read_hex(path):
    return np.array([int(x.strip(), 16) - (256 if int(x.strip(), 16) >= 128 else 0)
                     for x in path.read_text().splitlines() if x.strip()], dtype=np.int8)


def write_hex(path, values):
    path.write_text("".join(f"{int(v) & 0xff:02x}\n" for v in values))


def dense(x, weights, bias, layer):
    out = []
    total_shift = 31 + int(layer["shift"])
    for row, b in zip(weights, bias):
        acc = int(np.sum(x.astype(np.int64) * row.astype(np.int64))) + int(b)
        product = acc * int(layer["m0"])
        if total_shift > 0:
            value = (product + (1 << (total_shift - 1))) >> total_shift
        elif total_shift < 0:
            value = product << -total_shift
        else:
            value = product
        if layer["activation"] == "RELU":
            value = max(0, value)
        out.append(max(-128, min(127, value)))
    return np.array(out, dtype=np.int8)


def golden_for(vector, config, weights, biases):
    w_pos = b_pos = 0
    act = np.array(vector, dtype=np.int8)
    for layer in config["layers"]:
        count = layer["in_features"] * layer["out_features"]
        layer_w = weights[w_pos:w_pos + count].reshape(layer["out_features"], layer["in_features"])
        layer_b = biases[b_pos:b_pos + layer["out_features"]]
        act = dense(act, layer_w, layer_b, layer)
        w_pos += count
        b_pos += layer["out_features"]
    return act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compile", action="store_true", help="reuse flow/tb_flow.out")
    args = parser.parse_args()
    config = json.loads((FLOW / "model_config.json").read_text())
    vectors = json.loads((FLOW / "test_inputs.json").read_text())["test_vectors"]
    if not config.get("layers"):
        raise RuntimeError("model_config.json has no trained layer metadata; run train_and_export.py first")
    if not args.skip_compile:
        subprocess.run(["iverilog", "-g2012", "-s", "tb_flow", "-o", "flow/tb_flow.out", *RTL_SOURCES],
                       cwd=ROOT, check=True)

    weights, biases = read_hex(FLOW / "weight.hex"), read_hex(FLOW / "bias.hex")
    original_input = (FLOW / "input.hex").read_bytes()
    original_golden = (FLOW / "golden_output.hex").read_bytes()
    results = []
    try:
        for item in vectors:
            expected = golden_for(item["input_int8"], config, weights, biases)
            write_hex(FLOW / "input.hex", item["input_int8"])
            write_hex(FLOW / "golden_output.hex", expected)
            run = subprocess.run(["vvp", "flow/tb_flow.out"], cwd=ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            actual = read_hex(FLOW / "output.hex") if (FLOW / "output.hex").exists() else np.array([], dtype=np.int8)
            match = run.returncode == 0 and np.array_equal(actual, expected) and "FAILED:" not in run.stdout
            cycle_match = re.search(r"finished successfully in (\d+) clock cycles", run.stdout)
            cycles = int(cycle_match.group(1)) if cycle_match else None
            instruction_cycles = [
                {"id": int(match.group(1)), "opcode": int(match.group(2), 16),
                 "cycles": int(match.group(3))}
                for match in re.finditer(
                    r"\[TRACE\] instruction=(\d+) opcode=0x([0-9a-fA-F]+) cycles=(\d+)",
                    run.stdout,
                )
            ]
            results.append({"id": item["id"], "true_label": item["true_label"],
                            "expected": expected.tolist(), "actual": actual.tolist(),
                            "prediction": config["class_names"][int(np.argmax(expected))],
                            "match": match, "cycles": cycles,
                            "instruction_cycles": instruction_cycles})
            print(f"vector {item['id']}: {'PASS' if match else 'FAIL'}  cycles={cycles}")
    finally:
        (FLOW / "input.hex").write_bytes(original_input)
        (FLOW / "golden_output.hex").write_bytes(original_golden)

    matches = sum(r["match"] for r in results)
    correct = sum(r["prediction"] == r["true_label"] for r in results)
    cycle_values = [r["cycles"] for r in results if r["cycles"] is not None]
    opcode_names = {1: "LOAD", 2: "STORE", 3: "DENSE", 4: "ACT", 5: "END"}
    instruction_samples = {}
    for result in results:
        for sample in result["instruction_cycles"]:
            name = opcode_names.get(sample["opcode"], f"0x{sample['opcode']:02x}")
            instruction_samples.setdefault((sample["id"], name), []).append(sample["cycles"])
    lines = ["# Accelerator Performance Report", "", f"Model: `{config['model']}`", "",
             f"Topology: `{config['input_len']} -> 32 -> 16 -> {config['output_len']}`; "
             f"weights: {config['total_weights']}; biases: {config['total_biases']}", "",
             f"Float test accuracy: **{config.get('accuracy', {}).get('float_test', 0) * 100:.1f}%**; "
             f"INT8 test accuracy: **{config.get('accuracy', {}).get('int8_test', 0) * 100:.1f}%**", "",
             "| Vector | Ground truth | Prediction | Golden output | RTL output | Match | Cycles |",
             "|---:|---|---|---|---|---|---:|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['true_label']} | {r['prediction']} | `{r['expected']}` | `{r['actual']}` | {'PASS' if r['match'] else 'FAIL'} | {r['cycles'] or '—'} |")
    lines.extend(["", f"RTL match rate: **{matches}/{len(results)}**", f"Representative accuracy: **{correct}/{len(results)}**"])
    if cycle_values:
        lines.append(f"Average inference cycles: **{sum(cycle_values) / len(cycle_values):.1f}**")
    lines.extend(["", "## Average cycles by instruction", "",
                  "Instruction spans include fetch, decode, dispatch, execution, and retirement.", "",
                  "| Instruction | Opcode | Average cycles | Samples |",
                  "|---:|---|---:|---:|"])
    for (instruction_id, name), samples in sorted(instruction_samples.items()):
        lines.append(f"| {instruction_id} | {name} | {sum(samples) / len(samples):.1f} | {len(samples)} |")
    (FLOW / "perf_report.md").write_text("\n".join(lines) + "\n")
    print(f"RTL match rate: {matches}/{len(results)}")
    return 0 if matches == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
