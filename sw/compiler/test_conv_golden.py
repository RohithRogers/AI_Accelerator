"""
Python golden reference for a single-channel 3x3 convolution used by
 tb/tb_conv_engine.sv.

The hardware computes a 3x3 kernel over a 4x4 input with stride=1 and no
padding, producing a 2x2 output map. This script mirrors the expected INT8
result of the combined im2col + dense path.
"""

from typing import List


def conv_golden_3x3(input_img: List[int], kernel: List[int], bias: int = 0) -> List[int]:
    assert len(input_img) == 16, "expected 4x4 input"
    assert len(kernel) == 9, "expected 3x3 kernel"
    out = []
    for oy in range(2):
        for ox in range(2):
            acc = bias
            for ky in range(3):
                for kx in range(3):
                    iy = oy + ky
                    ix = ox + kx
                    idx = iy * 4 + ix
                    acc += input_img[idx] * kernel[ky * 3 + kx]
            if acc > 127:
                acc = 127
            elif acc < -128:
                acc = -128
            out.append(acc)
    return out


def conv_golden_3x3_multiout(input_img: List[int], kernels: List[List[int]],
                              biases: List[int]) -> List[int]:
    """Spatial-major [pixel][output_channel] layout used by the RTL."""
    assert len(kernels) == len(biases)
    per_channel = [conv_golden_3x3(input_img, kernel, bias)
                   for kernel, bias in zip(kernels, biases)]
    return [per_channel[channel][pixel]
            for pixel in range(4)
            for channel in range(len(kernels))]


if __name__ == "__main__":
    img = list(range(1, 17))
    kernel = [1, 1, 1,
              1, 1, 1,
              1, 1, 1]
    expected = conv_golden_3x3(img, kernel, 0)
    print(expected)
    assert expected == [54, 63, 90, 99], expected
    multiout = conv_golden_3x3_multiout(
        img,
        [kernel, [1, 0, 0, 0, 0, 0, 0, 0, 0]],
        [0, 1],
    )
    assert multiout == [54, 2, 63, 3, 90, 6, 99, 7], multiout
    print("=== CONV GOLDEN TEST PASSED ===")
