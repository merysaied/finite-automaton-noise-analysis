# Automaton Robustness Analysis

## About the Project

This project was developed as a **group project for the Automata course**. Its main objective is to investigate how noise and missing data affect the performance of a Decision Tree model and to explore the relationship between Decision Trees and Finite Automata.

The project uses the Titanic dataset to train a Decision Tree classifier, represents the tree as a finite automaton, and evaluates its robustness under different data corruption scenarios.

The project also includes a Streamlit dashboard for visualizing the model, automaton states, and experimental results.

## My Contribution

My main responsibility in this group project was improving the model training and noise experimentation pipeline in `train_model.py`.

The work I completed includes:

### 1. Data Preparation and Leakage Prevention
- Used `.copy()` when creating DataFrame subsets to avoid unintended modifications.
- Calculated the mean age using training data only.
- Applied the training mean to handle missing age values in both training and testing datasets.
- This prevents information from the test set from leaking into the training process.

### 2. Noise Generation
- Used a NumPy random generator (`rng`) for reproducible noise generation.
- Implemented independent noise generation for different features.
- Added Gaussian noise to age values.
- Restricted noisy age values to the valid range of 0–80.

### 3. Missing Data Simulation (Epsilon)
- Implemented the epsilon-loss scenario to simulate missing age information.
- Replaced affected age values with the mean age calculated from the training set.
- Saved the calculated mean age in the pickle artifact so the application can use the same imputation value.

### 4. Repeated Experiments and Statistical Evaluation
- Repeated each experiment 20 times to obtain more reliable results.
- Calculated and stored the mean accuracy and standard deviation for each noise scenario and noise level.
- Used these measurements to evaluate model stability under different noise conditions.

### 5. Noise Levels and Baseline
- Extended the noise levels to include 0.6, 0.8, and 0.9.
- Included the baseline accuracy (61.6%) as a reference for evaluating the model's performance under noisy conditions.

## Noise Scenarios

The project evaluates five scenarios:

| Scenario | Description |
|---|---|
| Age | Gaussian noise is added to age values. |
| Sex | Randomly flips encoded sex values. |
| Pclass | Randomly changes passenger ticket classes. |
| All Combined | Applies noise to age, sex, and passenger class. |
| Epsilon Loss | Simulates missing age data using mean imputation. |

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Pickle

## Project Structure

```text
automata/
│
├── train_model.py
├── app.py
├── train.csv
├── test.csv
├── model_artifacts.pkl
└── README.md
```

## How to Run the Project

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn streamlit matplotlib
```

### 2. Train the Model

```bash
python train_model.py
```

This generates `model_artifacts.pkl`, which contains the trained model, automaton information, noise experiment results, statistical measurements, and other values required by the application.

### 3. Run the Dashboard

```bash
python -m streamlit run app.py
```

The Streamlit dashboard will open in your browser.

## Project Purpose

The project demonstrates how a Decision Tree can be interpreted as a Finite Automaton and how its classification performance changes when its input data is affected by noise or missing information.

By repeating experiments and measuring both mean accuracy and standard deviation, the project provides a way to study not only model performance but also its stability and robustness.

## Academic Context

**Course:** Automata

**Project Type:** Group Project

**My Role:** Model training pipeline improvements, data preprocessing, noise simulation, repeated experimentation, and statistical result collection.

