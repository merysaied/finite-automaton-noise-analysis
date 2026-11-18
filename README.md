# Automaton Noise Analysis

## About the Project

This project was developed as a **group project for the Automata course**.

The project investigates how noise and missing data affect the performance of a Decision Tree model using the Titanic dataset. It also explores the relationship between Decision Trees and Finite Automata.

A Streamlit dashboard is used to visualize the results and analyze the effects of different noise levels.

## My Contribution

My contribution focused on the model training and noise analysis pipeline:

- Used `.copy()` to prevent unintended modifications to the original DataFrames.
- Calculated the mean age using training data only to prevent data leakage.
- Generated independent noise using a single NumPy random generator (`rng`).
- Clipped generated ages to the range of 0–80.
- Implemented missing-data handling (ε) using mean imputation.
- Saved the calculated mean age so it could also be used by the application.
- Repeated each experiment 20 times to obtain more reliable results.
- Calculated and saved the mean and standard deviation for each experiment.
- Added noise levels of 0.6, 0.8, and 0.9.
- Saved the trained model and experiment results in `model_artifacts.pkl`.

## Model Results

Results from the latest training run:

| Metric | Result |
|---|---:|
| Clean test accuracy | 80.45% |
| Majority-class baseline | 58.66% |
| Number of states | 15 |
| Accepting states | `q_3`, `q_11`, `q_13` |

## Noise Analysis

The project examines different noise scenarios:

- Age noise
- Sex noise
- Passenger class (Pclass) noise
- Combined noise
- Missing-data handling (ε)

Each experiment is repeated 20 times, and its mean and standard deviation are calculated.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Pickle

## How to Run

### 1. Install dependencies

```bash
pip install numpy pandas scikit-learn streamlit matplotlib
```

### 2. Train the model

```bash
python train_model.py
```

This generates the `model_artifacts.pkl` file.

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

## Academic Information

- **Course:** Automata
- **Project Type:** Group Project
- **Dataset:** Titanic
- **My Role:** Data preprocessing, noise generation, missing-data handling, repeated experiments, and statistical evaluation.

