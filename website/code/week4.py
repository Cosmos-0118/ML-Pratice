#: Unit 4 – Hidden Markov models

# %% Practice 1 | Implement HMM to predict sequential data — sleep-stage recognition
#: Infer Wake, N1, N2, N3 and REM from real EEG and eye-movement recordings in Sleep-EDF. Train on seven people and test on the eighth, repeating for all eight. Viterbi finds the most likely stage sequence; forward–backward estimates each stage's probability. This is whole-sequence recognition. Compare it with independent classification and a majority baseline; the benefit is modest and light sleep remains difficult.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Each row is a real, nonoverlapping 30-second signal segment.
# The 15 inputs are natural-log power in five frequency bands for three signals.
# Expert stage labels supervise training and evaluate predictions, never test decoding.
df = pd.read_csv("data/sleep_edf.csv")
df = df.sort_values(["subject", "epoch"]).reset_index(drop=True)
states = ["Wake", "N1", "N2", "N3", "REM"]
features = list(df.columns[4:])
subjects = list(df["subject"].unique())


def fit_hmm(training):
    # Fit every parameter using the training people only.
    labelled = training[training["stage"] >= 0]
    values = labelled[features].to_numpy()
    # Native integer width also works in the browser's 32-bit Python runtime.
    labels = labelled["stage"].to_numpy(dtype=int)
    centre = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = (values - centre) / scale
    means = np.empty((5, len(features)))
    variances = np.empty_like(means)
    for state in range(5):
        examples = scaled[labels == state]
        if len(examples) == 0:
            raise ValueError("Every sleep stage needs training examples")
        means[state] = examples.mean(axis=0)
        # A fixed variance floor prevents division by zero; no test tuning.
        variances[state] = np.maximum(examples.var(axis=0), 0.01)

    start_counts = np.ones(5)
    transition_counts = np.ones((5, 5))
    for subject, recording in training.groupby("subject", sort=True):
        sequence = recording["stage"].to_numpy()
        valid = np.flatnonzero(sequence >= 0)
        # Training gaps break a sequence; people are never joined together.
        breaks = np.flatnonzero(np.diff(valid) > 1) + 1
        for segment in np.split(valid, breaks):
            if len(segment) == 0:
                continue
            known = sequence[segment]
            start_counts[known[0]] += 1
            np.add.at(transition_counts, (known[:-1], known[1:]), 1)
    return {"centre": centre, "scale": scale, "means": means, "variances": variances,
            "start": start_counts / start_counts.sum(),
            "transition": transition_counts / transition_counts.sum(axis=1, keepdims=True),
            "prior": (np.bincount(labels, minlength=5) + 1) / (len(labels) + 5)}


def log_emissions(model, values):
    # A Gaussian density describes how each hidden stage emits sensor features.
    scaled = (values - model["centre"]) / model["scale"]
    difference = scaled[:, None, :] - model["means"][None, :, :]
    terms = np.log(2 * np.pi * model["variances"]) + difference ** 2 / model["variances"]
    return -0.5 * terms.sum(axis=2)


def viterbi(emissions, start, transition):
    # Log probabilities turn products into sums and avoid numerical underflow.
    if len(emissions) == 0:
        return np.empty(0, dtype=int)
    scores = np.log(start) + emissions[0]
    back = np.zeros(emissions.shape, dtype=int)
    log_transition = np.log(transition)
    for time in range(1, len(emissions)):
        candidates = scores[:, None] + log_transition
        back[time] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + emissions[time]
    path = np.empty(len(emissions), dtype=int)
    path[-1] = scores.argmax()
    for time in range(len(emissions) - 2, -1, -1):
        path[time] = back[time + 1, path[time + 1]]
    return path


def forward_backward(emissions, start, transition):
    # Sum over all possible paths, rather than keeping only Viterbi's best path.
    # logaddexp.reduce is a stable log(sum(exp(...))).
    if len(emissions) == 0:
        return np.empty(emissions.shape), 0.0
    log_transition = np.log(transition)
    forward = np.empty_like(emissions)
    backward = np.zeros_like(emissions)
    forward[0] = np.log(start) + emissions[0]
    for time in range(1, len(emissions)):
        forward[time] = emissions[time] + np.logaddexp.reduce(
            forward[time - 1, :, None] + log_transition, axis=0)
    for time in range(len(emissions) - 2, -1, -1):
        backward[time] = np.logaddexp.reduce(
            log_transition + emissions[time + 1] + backward[time + 1], axis=1)
    log_likelihood = np.logaddexp.reduce(forward[-1])
    log_posterior = forward + backward
    log_posterior -= np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
    return np.exp(log_posterior), log_likelihood


def decode_record(model, values):
    # Only signal inputs enter prediction: no expert labels or label-derived gaps.
    emissions = log_emissions(model, values)
    decoded = viterbi(emissions, model["start"], model["transition"])
    independent = (emissions + np.log(model["prior"])).argmax(axis=1)
    return decoded, independent


def evaluate(actual, predicted):
    # Every class gets equal weight in macro F1, even the rare light-sleep stage.
    matrix = np.zeros((5, 5), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    true_positive = matrix.diagonal()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    stage_f1 = np.divide(2 * true_positive, denominator,
                         out=np.zeros(5), where=denominator > 0)
    return np.mean(actual == predicted), stage_f1.mean(), stage_f1, matrix


# 2. Leave one person out, fit on the other seven, decode their entire recording.
# This supervised HMM learns from labelled training stages, without Baum–Welch.
records = []
fold_scores = []
fold_training_subjects = {}
for held_out in subjects:
    training = df[df["subject"] != held_out]
    test = df[df["subject"] == held_out].copy()
    model = fit_hmm(training)
    fold_training_subjects[held_out] = list(training["subject"].unique())
    test["decoded"], test["independent"] = decode_record(model, test[features].to_numpy())
    test["majority"] = model["prior"].argmax()
    # -1 is movement/unscored, excluded only when scoring predictions.
    scored = test[test["stage"] >= 0]
    row = {"Person": held_out}
    for column in ["majority", "independent", "decoded"]:
        accuracy, macro_f1, stage_f1, matrix = evaluate(
            scored["stage"].to_numpy(), scored[column].to_numpy())
        row[column + " accuracy"] = accuracy
        row[column + " macro F1"] = macro_f1
    fold_scores.append(row)
    records.append(test)
    # Always illustrate the first person; never select the best test result.
    if held_out == subjects[0]:
        example = test
        example_model = model

predictions = pd.concat(records, ignore_index=True)
scored = predictions[predictions["stage"] >= 0]
summary_rows = []
for column, title in zip(["majority", "independent", "decoded"],
                         ["Training-majority baseline", "Independent Gaussian", "HMM / Viterbi"]):
    accuracy, macro_f1, stage_f1, matrix = evaluate(scored["stage"].to_numpy(), scored[column].to_numpy())
    summary_rows.append({"Model": title, "Accuracy": accuracy, "Macro F1": macro_f1})
summary = pd.DataFrame(summary_rows).set_index("Model")
hmm_accuracy, hmm_macro_f1, hmm_stage_f1, confusion = evaluate(
    scored["stage"].to_numpy(), scored["decoded"].to_numpy())
print("Sleep-EDF: 8 people,", len(df), "signal segments,", len(scored), "scored segments")
print("Each person is tested with a model trained on the other seven.\n")
print("Pooled held-out results:\n", summary.round(4))
print("Per-person held-out results:\n", pd.DataFrame(fold_scores).round(3).to_string(index=False))
print("Per-stage HMM F1:\n", pd.Series(hmm_stage_f1, index=states).round(3))
print("Interpretation: modest improvement; N1 (light sleep) remains difficult.")
print("About 71% of these full recordings is Wake, which raises overall accuracy.")
print("Eight-person educational subset; offline decoding, not forecasting or clinical validation.")

# 3. A fixed clock-time view, selected without expert labels: hours 8–16 of SC400.
# Decode the full recording first. This display window does not change any scores.
hours = example["seconds"].to_numpy() / 3600
view = (hours >= 8) & (hours < 16)
posterior, example_log_likelihood = forward_backward(
    log_emissions(example_model, example[features].to_numpy()),
    example_model["start"], example_model["transition"])

# Graph 1: observed signals are different from the hidden sleep stages.
plt.figure(figsize=(10, 4))
for column, title in zip(["eeg_fpz_delta", "eeg_fpz_beta"],
                         ["EEG slow-wave power (0.5–4 Hz)", "EEG faster-wave power (16–30 Hz)"]):
    plt.plot(hours[view], example.loc[view, column], label=title, linewidth=1)
plt.title("Observed EEG features — held-out person SC400")
plt.xlabel("Hours from recording start (fixed 8–16 h view)")
plt.ylabel("Natural log of band power (µV²)")
plt.legend(fontsize=9)
plt.show()

# Graph 2: distinguish independent decisions from a decoded sequence.
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, column, title in zip(axes, ["stage", "independent", "decoded"],
                             ["Expert labels", "Independent Gaussian classification", "HMM / Viterbi sequence decoding"]):
    values = example[column].to_numpy(dtype=float)
    values[example["stage"].to_numpy() < 0] = np.nan
    ax.step(hours[view], values[view], where="post", linewidth=1)
    ax.set_yticks(range(5), states)
    ax.set_ylim(4.3, -0.3)
    ax.set_title(title)
axes[-1].set_xlabel("Hours from recording start — SC400 (fixed 8–16 h view)")
plt.show()

# Graph 3: what the other seven people taught the HMM about stage transitions.
fig, ax = plt.subplots(figsize=(7, 5))
image = ax.imshow(example_model["transition"], vmin=0, vmax=1, cmap="Blues")
ax.grid(False)
for row in range(5):
    for column in range(5):
        value = example_model["transition"][row, column]
        ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                color="white" if value > 0.6 else "black")
ax.set_xticks(range(5), states)
ax.set_yticks(range(5), states)
ax.set_xlabel("Next stage (30 seconds later)")
ax.set_ylabel("Current stage")
ax.set_title("Learned transitions — training people SC401–SC407\nRounded 0.00 values remain nonzero after smoothing")
fig.colorbar(image, ax=ax, label="Transition probability")
plt.show()

# Graph 4: forward–backward probabilities, using past and future observations.
fig, ax = plt.subplots(figsize=(10, 4))
image = ax.imshow(posterior[view].T, aspect="auto", interpolation="nearest",
                  extent=[hours[view][0], hours[view][-1] + 30 / 3600, 4.5, -0.5],
                  vmin=0, vmax=1, cmap="viridis")
ax.grid(False)
ax.set_yticks(range(5), states)
ax.set_xlabel("Hours from recording start — held-out person SC400")
ax.set_title("Forward–backward stage probabilities (whole sequence)\nModel probabilities have not been calibrated")
fig.colorbar(image, ax=ax, label="Model probability")
plt.show()

# Graph 5: all held-out people, with rare stages visible in their own rows.
fig, ax = plt.subplots(figsize=(7, 5))
proportions = confusion / confusion.sum(axis=1, keepdims=True)
image = ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues")
ax.grid(False)
for row in range(5):
    for column in range(5):
        value = proportions[row, column]
        ax.text(column, row, f"{value:.0%}\n({confusion[row, column]})", ha="center", va="center",
                fontsize=9, color="white" if value > 0.6 else "black")
ax.set_xticks(range(5), states)
ax.set_yticks(range(5), states)
ax.set_xlabel("HMM prediction")
ax.set_ylabel("Expert label")
ax.set_title("All 8 held-out people — confusion matrix\nPercent within each true stage; counts in parentheses")
fig.colorbar(image, ax=ax, label="Fraction of true stage")
plt.show()
