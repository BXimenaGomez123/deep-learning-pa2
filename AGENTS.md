# Repository guide

## Structure

- `src/pa2/` — package source code:
  - `synthetic_dataset.py` — synthetic video generator and frame/track datasets.
  - `mot17_dataset.py` — MOT17 frame and track datasets.
  - `box_corruption.py` — utilities for simulating missed, noisy, and false-positive track boxes.
  - `gru_model.py` — generic GRU sequence model; independent of dataset and box formats.
  - `metrics.py` — IoU matching, identity switches, and IDF1 utilities.
  - `plotting.py` — video and box animation helper.
  - `device.py` — PyTorch device selection.
- `tests/` — pytest tests, organized around package modules. `utils.py` contains shared test helpers.
- `scripts/` — runnable examples and visualization scripts.
- `data/` — local MOT17 images and labels; dataset files may not be present in every checkout.
- `justfile` — project tasks, including dataset downloads and checks.

## Project commands

Read `justfile` before running project tasks so the available recipes and their behavior are clear. Use `just` recipes rather than invoking test or type-check commands directly:

- `just test` — run the test suite.
- `just typecheck` — run the type checker.
- `just all_checks` — run both checks.

Dataset download recipes in `justfile` access the network and write under `data/`; only run them when needed.

## Implementation guidance

- Keep code as simple as possible; don't add features not asked for beforehand.
- Try to keep most of project's logic under `src/pa2/` instead of `scripts/`.
- Keep code as decoupled as possible.
- Keep models independent of dataset loading and tracking/association logic.
- When writing scripts, don't use `argparse` and have most of the functionalities as `import`s.
- Preserve the established dataset tensor shapes and dtypes unless a task explicitly changes that interface.
- Add or update tests under `tests/` for behavior changes. Script functionality doesn't need to be tested.
