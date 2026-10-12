# Original research code (kept for reference)

These notebooks and scripts are the first version of the project (May 2025): k-NN and SVM
tuned by a genetic algorithm (DEAP), run in Google Colab on PROMISE releases.

**Their scores are not valid estimates.** They standardise the data and apply SMOTE to the
whole dataset *before* cross-validation, so synthetic near-copies of each test class end up in
the training folds (and synthetic rows are tested). This is why they report F1 scores of
up to 0.98 (above 0.93 on 6 of the 10 projects), mostly with 1-nearest-neighbour models. The GA is also scored on the same folds that
are reported, which adds selection bias.

The rewritten package (`bugpredictor/`) fits every step inside the training folds and
evaluates the GA with nested cross-validation. `experiments/run_all.py` re-runs the original
protocol next to the corrected one: see section 1 of `results/results.md`.

The files are left as they were, except that empty placeholder files were removed and the
pickled models were deleted (a pickle can run arbitrary code when loaded, and these models
were trained with the leaky protocol).
