#: Unit 4 – Hidden Markov models

# %% Practice 1 | Implement HMM to predict the sequential data
#: The synthetic weather regimes persist across consecutive days. The hidden state is wet or dry; the observation is the temperature range. Learn probabilities using the known states in 2012–2014, then hide the states and decode 2015 with Viterbi. This is supervised HMM parameter estimation. Compare the decoded sequence with the true states and forecast future wet/dry probabilities.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/weather_regimes.csv").sort_values("date")

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
print("Viterbi accuracy on 2015:", round(accuracy, 3))
majority_state = train["state"].value_counts().idxmax()
baseline = (test["state"] == majority_state).mean()
print("Always predict the training majority (" + majority_state + "):", round(baseline, 3))
print()
print("Actual (rows) vs guessed (columns):")
print(pd.crosstab(test["state"], test["guess"]))
print()

# 3. Forecast: suppose today is wet. How likely is a wet day in each of the next 5 days?
p = np.array([0.0, 1.0])   # [P(dry), P(wet)] today
for day in range(1, 6):
    p = p @ transition.loc[states, states].values
    print("Day", day, "-> P(wet) =", round(p[1], 2))

# Plot both state sequences: dry days stay visible even with zero precipitation.
first = test.head(120)
fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True, layout="constrained")
for axis, column, title in zip(axes, ["state", "guess"], ["Actual wet/dry states", "HMM decoded states"]):
    axis.step(range(1, len(first) + 1), (first[column] == "wet").astype(int), where="mid")
    axis.set_yticks([0, 1], labels=["dry", "wet"])
    axis.set_ylim(-0.15, 1.15)
    axis.set_title(title)
axes[-1].set_xlabel("day of 2015 (first 120 days)")
plt.show()
