#: Unit 1 – Introduction

# %% Practice 1 | Devise a program to import, load and view dataset
#: Load the Seattle weather CSV file into a pandas DataFrame (a table), then look at its size, its columns and its first and last rows.
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("data/seattle_weather.csv")

# Size of the table
print("Shape (rows, columns):", df.shape)
print()

# Column names, data types and missing values
df.info()
print()

# First 10 rows and last 5 rows
print("First 10 rows:")
print(df.head(10))
print()
print("Last 5 rows:")
print(df.tail())

# %% Practice 2 | Create a program to display the summary and statistics of the dataset
#: describe() gives count, mean, standard deviation, minimum, quartiles and maximum of every number column. We also count missing values and weather types, and draw a few plots.
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/seattle_weather.csv")

# Summary statistics of the number columns
print(df.describe().round(2))
print()

# Missing values in each column
print("Missing values in each column:")
print(df.isnull().sum())
print()

# How many days of each weather type
print("Days of each weather type:")
print(df["weather"].value_counts())
print()

# A few statistics of one column
print("Mean of temp_max:  ", round(df["temp_max"].mean(), 2))
print("Median of temp_max:", df["temp_max"].median())
print("Std of temp_max:   ", round(df["temp_max"].std(), 2))
print()

# Correlation: +1 = rise together, -1 = one rises when the other falls, 0 = no link
print("Correlation between the number columns:")
print(df.corr(numeric_only=True).round(2))

# Plot 1: histogram of every number column
df.hist(bins=30, figsize=(9, 5.6))
plt.show()

# Plot 2: number of days of each weather type
plt.figure(figsize=(9, 4.5))
df["weather"].value_counts().plot(kind="bar")
plt.title("Days of each weather type")
plt.ylabel("days")
plt.show()

# Plot 3: average temp_max in each month
df["month"] = pd.to_datetime(df["date"]).dt.month
plt.figure(figsize=(9, 4.5))
df.groupby("month")["temp_max"].mean().plot(kind="bar", color="orange")
plt.title("Average temp_max in each month")
plt.ylabel("°C")
plt.show()
