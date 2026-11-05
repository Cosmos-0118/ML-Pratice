"""Glue for the Unit 4 explorer page. It runs after the model code of week4.py (everything above the
"# 2. Leave one person out" line) in the same Python namespace, so the page uses exactly the same
fit_hmm, log_emissions, viterbi, forward_backward and run_fold that the practice program shows.
It adds no modelling of its own: it only packages one fold's results as JSON for the charts."""
import json

import numpy as np


def explore(held_out):
    """Train on the other seven people, decode `held_out`, and return everything the page draws."""
    test, model, row, training_subjects = run_fold(held_out)
    emissions = log_emissions(model, test[features].to_numpy())
    posterior, log_likelihood = forward_backward(emissions, model["start"], model["transition"])
    actual = test["stage"].to_numpy()
    scored = actual >= 0
    models = {}
    for column in ["majority", "independent", "decoded"]:
        accuracy, macro_f1, stage_f1, matrix = evaluate(actual[scored], test[column].to_numpy()[scored])
        models[column] = {"accuracy": float(accuracy), "macro_f1": float(macro_f1),
                          "stage_f1": stage_f1.tolist(), "confusion": matrix.tolist()}
    trained = df[(df["subject"] != held_out) & (df["stage"] >= 0)]
    return json.dumps({
        "subject": held_out,
        "states": states,
        "features": features,
        "training_subjects": training_subjects,
        "training_segments": int(len(trained)),
        "training_stage_counts": np.bincount(trained["stage"].to_numpy(dtype=int), minlength=5).tolist(),
        "segments": int(len(test)),
        "scored_segments": int(scored.sum()),
        "expert": actual.tolist(),
        "majority": test["majority"].tolist(),
        "independent": test["independent"].tolist(),
        "decoded": test["decoded"].tolist(),
        "posterior": np.round(posterior, 3).tolist(),
        "eeg_delta": np.round(test["eeg_fpz_delta"].to_numpy(), 2).tolist(),
        "eeg_beta": np.round(test["eeg_fpz_beta"].to_numpy(), 2).tolist(),
        "models": models,
        "transition": model["transition"].tolist(),
        "signature": model["means"].tolist(),
        "log_likelihood": float(log_likelihood),
        "row": {key: (value if isinstance(value, str) else float(value)) for key, value in row.items()},
    })
