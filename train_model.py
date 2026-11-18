"""
Robustness analysis of a decision tree used as a finite automaton.

Pipeline:
  1. Load the Titanic data and split it into train/test.
  2. Train a shallow decision tree (each node = a state of the automaton).
  3. Corrupt the test inputs with different noise scenarios and measure
     how the accuracy degrades (repeated several times for stable results).
  4. Save everything the Streamlit app needs into a pickle file.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42                    # base seed for reproducibility
N_REPEATS = 20               # repetitions per (scenario, noise level)
AGE_MIN, AGE_MAX = 0, 80     # valid age range after adding noise
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ----------------------------------------------------------------------
# Step 1: Load and prepare the data
# ----------------------------------------------------------------------
print("Step 1: Loading and preparing data...")
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    # Fallback: download the same dataset if the local file is missing
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Keep only the columns we need; .copy() avoids SettingWithCopyWarning
df = df[["Survived", "Pclass", "Sex", "Age"]].copy()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # encode as numbers

X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

# Split first, then impute with the TRAIN mean only (avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
age_mean = X_train["Age"].mean()
X_train = X_train.copy()
X_test = X_test.copy()
X_train["Age"] = X_train["Age"].fillna(age_mean)
X_test["Age"] = X_test["Age"].fillna(age_mean)

# ----------------------------------------------------------------------
# Step 2: Train the decision tree (our automaton)
# ----------------------------------------------------------------------
print("Step 2: Training the decision tree (automaton)...")
clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
clf.fit(X_train, y_train)

# Automaton view of the tree:
#   Q  = all tree nodes (states), q0 = root node (start state)
#   F  = leaf nodes that predict "survived" (accepting states)
num_nodes = clf.tree_.node_count
Q_states = [f"q_{i}" for i in range(num_nodes)]
q0_start = "q_0"

F_accept = []
for i in range(num_nodes):
    if clf.tree_.children_left[i] == -1:  # leaf node
        if np.argmax(clf.tree_.value[i]) == 1:  # leaf predicts "survived"
            F_accept.append(f"q_{i}")

# Accuracy of a trivial model that always predicts the majority class
baseline_acc = max(y_test.mean(), 1 - y_test.mean())


# ----------------------------------------------------------------------
# Noise functions (each one takes its own random generator `rng`,
# so the noise applied to different features is independent)
# ----------------------------------------------------------------------
def corrupt_age(X_data, level, rng):
    """Add Gaussian noise proportional to the age, then clip to a valid range."""
    X_noisy = X_data.copy()
    if level > 0:
        noise = level * rng.standard_normal(len(X_noisy)) * X_noisy["Age"]
        X_noisy["Age"] = (X_noisy["Age"] + noise).clip(AGE_MIN, AGE_MAX)
    return X_noisy


def corrupt_sex(X_data, level, rng):
    """Flip the sex value (0 <-> 1) for a random fraction `level` of passengers."""
    X_noisy = X_data.copy()
    if level > 0:
        mask = rng.random(len(X_noisy)) < level
        X_noisy.loc[mask, "Sex"] = 1 - X_noisy.loc[mask, "Sex"]
    return X_noisy


def corrupt_pclass(X_data, level, rng):
    """Replace the ticket class with a random class for a fraction `level` of passengers."""
    X_noisy = X_data.copy()
    if level > 0:
        mask = rng.random(len(X_noisy)) < level
        random_classes = rng.choice([1, 2, 3], size=len(X_noisy))
        X_noisy.loc[mask, "Pclass"] = random_classes[mask]
    return X_noisy


def corrupt_epsilon(X_data, level, rng):
    """Simulate missing data (epsilon): lost ages are replaced by the train mean age."""
    X_noisy = X_data.copy()
    if level > 0:
        mask = rng.random(len(X_noisy)) < level
        X_noisy.loc[mask, "Age"] = age_mean
    return X_noisy


def corrupt_all(X_data, level, rng):
    """Apply age noise, sex flips and class changes one after another."""
    X_noisy = corrupt_age(X_data, level, rng)
    X_noisy = corrupt_sex(X_noisy, level, rng)
    X_noisy = corrupt_pclass(X_noisy, level, rng)
    return X_noisy


# Scenario name -> noise function (names are used as keys in the app)
SCENARIOS = {
    "Age": corrupt_age,
    "Sex": corrupt_sex,
    "Pclass": corrupt_pclass,
    "All_Combined": corrupt_all,
    "Epsilon_Loss": corrupt_epsilon,
}

# ----------------------------------------------------------------------
# Step 3: Run the noise experiments
# ----------------------------------------------------------------------
print(f"Step 3: Noise analysis ({N_REPEATS} repeats per level)...")
results_mean = {name: [] for name in SCENARIOS}  # mean accuracy per level
results_std = {name: [] for name in SCENARIOS}   # standard deviation per level

for lvl in NOISE_LEVELS:
    for name, corrupt_fn in SCENARIOS.items():
        accs = []
        for rep in range(N_REPEATS):
            # A different seed for every repetition, same seeds for every scenario
            rng = np.random.default_rng(SEED + rep)
            X_noisy = corrupt_fn(X_test, lvl, rng)
            accs.append(accuracy_score(y_test, clf.predict(X_noisy)))
        results_mean[name].append(float(np.mean(accs)))
        results_std[name].append(float(np.std(accs)))

# ----------------------------------------------------------------------
# Step 4: Save the artifacts for the Streamlit app
# ----------------------------------------------------------------------
print("Step 4: Saving artifacts...")
artifacts = {
    "model": clf,
    "noise_levels": NOISE_LEVELS,
    "results": results_mean,        # mean accuracy per scenario and level
    "results_std": results_std,     # standard deviation per scenario and level
    "n_repeats": N_REPEATS,
    "seed": SEED,
    "age_mean": age_mean,           # used by the app for the epsilon scenario
    "age_range": (AGE_MIN, AGE_MAX),
    "baseline_acc": baseline_acc,   # majority-class accuracy on the test set
    "X_test": X_test,
    "y_test": y_test,
    "Q": Q_states,
    "F": F_accept,
    "q0": q0_start,
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# Short summary
print(f"Test accuracy (clean): {results_mean['Age'][0] * 100:.2f}%")
print(f"Majority-class baseline: {baseline_acc * 100:.2f}%")
print(f"States: {num_nodes}, accepting: {F_accept}")
print("Done: 'model_artifacts.pkl' saved.")