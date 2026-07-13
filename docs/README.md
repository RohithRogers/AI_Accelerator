# Design docs

- [RTL implementation guide](implementation.md) — architecture, interfaces, coding order, and verification milestones.
- [RTL directory guide](../rtl/README.md) — intended source layout and module ownership.

The Python assembler at `sw/compiler/compiler.py` is the executable source of truth for instruction encoding. Keep the RTL decoder bit-for-bit compatible with it.
