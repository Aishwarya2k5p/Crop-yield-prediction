import pandas as pd

df = pd.read_csv("crop_yield.csv")

# Get unique values for crops, seasons, and states
crops = sorted(df["Crop"].dropna().unique())
seasons = sorted(df["Season"].dropna().unique())
states = sorted(df["State"].dropna().unique())

print("Crops:", crops)
print("Seasons:", seasons)
print("States:", states)
