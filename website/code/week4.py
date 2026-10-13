#: Unit 4 – Hidden Markov models

# %% Practice 1 | Implement HMM to predict the sequential data
#: Use real Seattle weather. Learn wet/dry probabilities from 2012–2014 and decode 2015 using Viterbi. Decoding uses the whole observed temperature sequence. Evaluate next-day forecasts separately using yesterday's known state, and compare them with simple baselines.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load days in order. Rain or snow makes a day wet.
df = pd.read_csv("data/seattle_weather.csv").sort_values("date")
df["state"] = np.where(df["precipitation"] > 0, "wet", "dry")
df["temp_range"] = df["temp_max"] - df["temp_min"]
df["seen"] = pd.cut(df["temp_range"], [-1, 5, 9, 100], labels=["small", "medium", "large"])
train = df[df["date"] < "2015-01-01"]
test = df[df["date"] >= "2015-01-01"].copy()
states = ["dry", "wet"]
observations = ["small", "medium", "large"]

# 2. Count states, transitions and observations using training days only.
# Add 1 so every possible event has a nonzero probability.
start_counts = train["state"].value_counts().reindex(states, fill_value=0) + 1
start = start_counts / start_counts.sum()
transition_counts = pd.crosstab(train["state"], train["state"].shift(-1))
transition_counts = transition_counts.reindex(index=states, columns=states, fill_value=0) + 1
transition = transition_counts.div(transition_counts.sum(axis=1), axis=0)
emission_counts = pd.crosstab(train["state"], train["seen"])
emission_counts = emission_counts.reindex(index=states, columns=observations, fill_value=0) + 1
emission = emission_counts.div(emission_counts.sum(axis=1), axis=0)
print("Starting probabilities:\n", start.round(2))
print("Transitions: row = today, column = tomorrow\n", transition.round(2))
print("Observations: row = state, column = temperature range\n", emission.round(2))

# 3. Viterbi keeps the best path ending in dry and the best path ending in wet.
# Log probabilities turn multiplication into addition and avoid tiny-number errors.
def viterbi(seen):
    if len(seen) == 0:
        return []
    scores = {}
    paths = {}
    for state in states:
        scores[state] = np.log(start[state]) + np.log(emission.loc[state, seen[0]])
        paths[state] = [state]

    for observation in seen[1:]:
        new_scores = {}
        new_paths = {}
        for state in states:
            from_dry = scores["dry"] + np.log(transition.loc["dry", state])
            from_wet = scores["wet"] + np.log(transition.loc["wet", state])
            if from_wet > from_dry:
                previous = "wet"
                best_score = from_wet
            else:
                previous = "dry"
                best_score = from_dry
            new_scores[state] = best_score + np.log(emission.loc[state, observation])
            new_paths[state] = paths[previous] + [state]
        scores = new_scores
        paths = new_paths

    if scores["wet"] > scores["dry"]:
        return paths["wet"]
    return paths["dry"]

# 4. Decode the test sequence, then compare it with the real states.
test["guess"] = viterbi(list(test["seen"]))
accuracy = (test["guess"] == test["state"]).mean()
majority_state = train["state"].value_counts().idxmax()
baseline = (test["state"] == majority_state).mean()
print("Viterbi decoding accuracy:", round(accuracy, 3))
print("Training-majority baseline:", round(baseline, 3))
print("Actual rows, decoded columns:\n", pd.crosstab(test["state"], test["guess"]))

# 5. Forecast each day before observing it, using only yesterday's state.
previous_states = df["state"].shift(1).loc[test.index]
forecast_predictions = []
for yesterday in previous_states:
    if transition.loc[yesterday, "wet"] > 0.5:
        forecast_predictions.append("wet")
    else:
        forecast_predictions.append("dry")
test["forecast"] = forecast_predictions
forecast_accuracy = (test["forecast"] == test["state"]).mean()
persistence_accuracy = (previous_states.values == test["state"].values).mean()
print("Rolling next-day forecast accuracy:", round(forecast_accuracy, 3))
print("Yesterday's state again (persistence):", round(persistence_accuracy, 3))

# Graph 1: three simple timelines. 0 means dry; 1 means wet.
first = test.head(120)
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
columns = ["state", "guess", "forecast"]
titles = ["Actual states", "Viterbi decoding (whole sequence)", "Next-day forecast (yesterday only)"]
for i in range(3):
    axes[i].step(range(1, 121), (first[columns[i]] == "wet").astype(int), where="mid")
    axes[i].set_yticks([0, 1], ["dry", "wet"])
    axes[i].set_ylim(-0.15, 1.15)
    axes[i].set_title(titles[i])
axes[2].set_xlabel("Day of 2015 (first 120 days)")
plt.show()

# Graph 2: suppose today is wet. Forecast five days with no new observations.
wet_probability = 1.0
future_probabilities = [wet_probability]
for day in range(1, 6):
    dry_probability = 1 - wet_probability
    wet_probability = dry_probability * transition.loc["dry", "wet"] + wet_probability * transition.loc["wet", "wet"]
    future_probabilities.append(wet_probability)
plt.figure(figsize=(9, 5))
plt.plot(range(6), future_probabilities, "o-")
plt.xlabel("Days ahead (0 = today)")
plt.ylabel("Probability of a wet day")
plt.ylim(0, 1)
plt.title("Five-day forecast, starting from a wet day")
plt.show()
