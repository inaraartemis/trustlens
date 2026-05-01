import pandas as pd

fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true])
df = df[["text", "label"]]

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("../data/combined.csv", index=False)

print("✅ Dataset created successfully!")
print("Shape:", df.shape)