#: Unit 4 – Hidden Markov models

# %% Practice 1 | Implement HMM to predict the sequential data
#: Use real Seattle daily weather. The hidden state is wet or dry; the observation is the temperature range. Estimate the HMM using known states in 2012–2014, then decode 2015 with Viterbi. This is supervised parameter estimation. Decoding uses the entire observed sequence; it is not a forecast. Separately evaluate rolling next-day transition forecasts using only yesterday's known state, and compare with persistence and majority baselines. Temperature alone is a weak signal, so errors are expected.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/seattle_weather.csv").sort_values("date")

# Hidden state: wet or dry
df["state"] = np.where(df["precipitation"] > 0, "wet", "dry")

# Observation: the temperature range of the day, as small / medium / large
df["temp_range"] = df["temp_max"] - df["temp_min"]
df["seen"] = pd.cut(df["temp_range"], bins=[-1, 5, 9, 100], labels=["small", "medium", "large"])

# Learn from 2012-2014, test on 2015
train = df[df["date"] < "2015-01-01"]
test = df[df["date"] >= "2015-01-01"].copy()
states = ["dry", "wet"]
observations = ["small", "medium", "large"]

# 1. Learn the HMM by counting
# Initial prior for an unseen sequence: training-state frequencies.
# Add one to every count so an unseen transition/observation never has probability 0.
start_counts = train["state"].value_counts().reindex(states, fill_value=0) + 1
start = start_counts / start_counts.sum()
# Transition probability: P(tomorrow's state | today's state)
transition_counts = pd.crosstab(train["state"], train["state"].shift(-1))
transition_counts = transition_counts.reindex(index=states, columns=states, fill_value=0) + 1
transition = transition_counts.div(transition_counts.sum(axis=1), axis=0)
# Emission probability: P(what we see | state)
emission_counts = pd.crosstab(train["state"], train["seen"])
emission_counts = emission_counts.reindex(index=states, columns=observations, fill_value=0) + 1
emission = emission_counts.div(emission_counts.sum(axis=1), axis=0)

print("Start probabilities:")
print(start.round(2))
print()
print("Transition probabilities (row = today, column = tomorrow):")
print(transition.round(2))
print()
print("Emission probabilities (row = state, column = what we see):")
print(emission.round(2))
print()


# 2. Viterbi: the most likely sequence of hidden states
def viterbi(seen):
    if not seen:
        return []
    # prob[s] = log probability of the best path ending in state s.
    # Adding log probabilities avoids numerical underflow on long sequences.
    # path[s] = that path
    prob = {}
    path = {}
    for s in states:
        prob[s] = np.log(start[s]) + np.log(emission.loc[s, seen[0]])
        path[s] = [s]

    for day in range(1, len(seen)):
        new_prob = {}
        new_path = {}
        for s in states:
            # Which state yesterday leads to s with the highest probability?
            best_prev = "dry"
            best_value = -np.inf
            for prev in states:
                value = prob[prev] + np.log(transition.loc[prev, s])
                if value > best_value:
                    best_value = value
                    best_prev = prev
            new_prob[s] = best_value + np.log(emission.loc[s, seen[day]])
            new_path[s] = path[best_prev] + [s]
        prob = new_prob
        path = new_path

    # Return the path that ends with the higher probability
    if prob["wet"] > prob["dry"]:
        return path["wet"]
    return path["dry"]


test["guess"] = viterbi(list(test["seen"]))
accuracy = (test["guess"] == test["state"]).mean()
print("Viterbi decoding accuracy on 2015 (uses the whole observation sequence):", round(accuracy, 3))
majority_state = train["state"].value_counts().idxmax()
baseline = (test["state"] == majority_state).mean()
print("Always predict the training majority (" + majority_state + "):", round(baseline, 3))
print()
print("Actual (rows) vs guessed (columns):")
print(pd.crosstab(test["state"], test["guess"]))
print()

# 3. Rolling forecasts, before seeing today's observation or precipitation.
# Yesterday's state is revealed as each day finishes; this is not a year-ahead forecast.
previous_states = df["state"].shift(1).loc[test.index]
forecast_from_actual = transition.loc[previous_states, states].to_numpy()
forecast_predictions = np.array(states)[forecast_from_actual.argmax(axis=1)]
forecast_accuracy = (forecast_predictions == test["state"]).mean()
persistence_accuracy = (previous_states.values == test["state"].values).mean()
brier_score = np.mean((forecast_from_actual[:, 1] - (test["state"] == "wet")) ** 2)
print("Rolling next-day forecast accuracy:", round(forecast_accuracy, 3))
print("Yesterday's state again (persistence):", round(persistence_accuracy, 3))
print("Training-majority baseline:", round(baseline, 3))
print("Forecast Brier score (lower is better):", round(brier_score, 3))

# Plot both state sequences: dry days stay visible even with zero precipitation.
test["forecast"] = forecast_predictions
first = test.head(120)
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout="constrained")
for axis, column, title in zip(axes, ["state", "guess", "forecast"], ["Actual wet/dry states", "Viterbi decoded states (whole sequence)", "Rolling next-day forecasts (yesterday's state only)"]):
    axis.step(range(1, len(first) + 1), (first[column] == "wet").astype(int), where="mid")
    axis.set_yticks([0, 1], labels=["dry", "wet"])
    axis.set_ylim(-0.15, 1.15)
    axis.set_title(title)
axes[-1].set_xlabel("day of 2015 (first 120 days)")
plt.show()

# Inspect the learned transition and observation models.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, table, title in zip(axes, [transition, emission], ["Transitions: tomorrow given today", "Emissions: temperature range given state"]):
    image = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), labels=list(table.columns))
    ax.set_yticks(range(len(states)), labels=states)
    ax.set_ylabel("Today's state")
    ax.set_title(title)
    ax.grid(False)
    for row in range(len(states)):
        for col in range(len(table.columns)):
            value = table.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.6 else "black")
fig.colorbar(image, ax=axes, label="Probability")
plt.show()

# Multi-day forecasts have no new observations: propagate both starting states.
fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
for initial in range(len(states)):
    p = np.eye(len(states))[initial]
    wet_probabilities = [p[1]]
    for day in range(1, 6):
        p = p @ transition.loc[states, states].values
        wet_probabilities.append(p[1])
    print(f"Next five P(wet), starting {states[initial]}:", np.round(wet_probabilities[1:], 3))
    ax.plot(range(6), wet_probabilities, "o-", label=f"Today is {states[initial]}")
ax.set(xlabel="Days ahead (0 = today)", ylabel="Probability of wet weather", ylim=(0, 1), xticks=range(6), title="Transition-only forecasts converge as the starting state is forgotten")
ax.legend()
plt.show()
