import pandas as pd

df = pd.read_csv("data/creditcard.csv")
fraud = df[df["Class"] == 1].sample(n=25, random_state=42)
legit = df[df["Class"] == 0].sample(n=475, random_state=42)

# Mix them together and shuffle so frauds aren't all at the top.
demo = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
demo.to_csv("data/demo_transactions.csv", index=False)
print(f"Saved {len(demo)} rows to data/demo_transactions.csv")