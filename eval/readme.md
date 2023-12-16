# Eval

## Directory Structure

- `eval-scripts`: contains the scripts to perform evaluation for this project. __Follow the readme therein__ (`eval/eval-scripts/readme.md`) to set up the test environment and run evaluations.

- `generations`: holding place for the generation outputs that the `infer` scripts produce for the various experiments. These outputs are the subjects of the evaluation/grading/testing scripts contained in the `eval-scripts` subdirectory.

- `results`: Saves the results (pass@K performance metrics) captured from each experiment, specifically, pass@{1,...,5}. If not otherwise noted in the filename, assume that the metrics are for 650 questions (corresponding to the first 650 introductory-level questions in the test split of the APPS dataset). These results are produced by running the evaluation/test scripts, following the instructions in `eval/eval-scripts/readme.md`.

## Infrastructure and Setup
See `eval/eval-scripts/readme.md` for how to set up the code-execution sandbox and run tests.
