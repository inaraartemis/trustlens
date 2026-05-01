import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("../data/X_test.npy", X_test)
np.save("../data/y_test.npy", y_test)

print("✅ Test data saved")