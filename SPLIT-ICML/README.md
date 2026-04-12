# SPLIT-ICML

This directory vendors the local `split` package used by the current MSPLIT
experiments.

## Active Selectors

The current codebase maintains two MSPLIT selector variants:

- `linear`
  `split/src/libgosdt/src/msplit_linear.cpp`
- `nonlinear`
  `split/src/libgosdt/src/msplit_nonlinear.cpp`

The nonlinear selector is the default active path for current experiments.

## Important Subdirectories

- `split/`
  Active Python package, C++ solver sources, bindings, and tests.

## Local Development

Build the extension from `split/` and add the resulting build directory plus
`split/src` to `PYTHONPATH` when running scripts from the repository root.
