# DEPRECATED — v1 Outputs (DO NOT USE)

These outputs were generated with buggy code and are preserved only for
reference. See `CHANGELOG.md` in the project root for details.

## Known issues in these results:

1. **ground_truth_flooded** was the FINAL simulation state for all entries
   (not per-step) — all detection metrics are unreliable
2. **Config mutation** — shallow copy caused parameter bleed between episodes
   during data generation
3. **Consensus feature** — was synthetically generated with `tanh()` heuristic
   instead of simulating actual sensors

## Regeneration

Run the pipeline described in `CHANGELOG.md` to produce corrected outputs
in the parent `outputs/` directory.
