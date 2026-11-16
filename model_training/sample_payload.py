import json
import sys

import pandas as pd

# Usage: python model_training/sample_payload.py 1   (a fraud row)
#        python model_training/sample_payload.py 0   (a legitimate row)
label = int(sys.argv[1]) if len(sys.argv) > 1 else 1
df = pd.read_csv("data/creditcard.csv")
df["Hour"] = (df["Time"] // 3600) % 24
row = df[df["Class"] == label].iloc[0].drop(["Time", "Class"]).to_dict()
row["Hour"] = int(row["Hour"])
print(json.dumps(row))