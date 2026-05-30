# Simple Modeling Lab

A modular Python repository showcasing how to implement and optimize different types of machine learning models from scratch using NumPy. The project is split into three standalone files (`lin_model.py`, `sin_model.py`, and `poly_model.py`) to handle varying data distributions.

## Core Modeling Workflow

For each data type, the modeling process follows a strict 4-step pipeline:

1. **Data Exploration:** Read the input dataset and generate scatter plots for initial visual analysis.
2. **Mathematical Setup:** Implement the custom model prediction logic ($\hat{y}$) and derive its corresponding analytical L2 loss gradient vector ($\nabla_{\vec{\theta}}L$).
3. **Optimization:** Apply a custom-built Gradient Descent loop to iteratively minimize the Mean Squared Error (MSE) and learn the optimal parameter weights ($\vec{\theta}$).
4. **Evaluation:** Plot the final trained model's prediction curve directly against the original data observations to visually verify convergence.

---

## Model Components

### 1. Linear Model (`lin_model.py`)
* **Mathematical Form:** $\hat{y} = \theta_0 + \theta_1 x$
* **Characteristics:** Minimizes standard Mean Squared Error on a globally convex loss surface, guaranteeing an optimal baseline intercept and slope solution.

### 2. Sinusoidal Model (`sin_model.py`)
* **Mathematical Form:** $\hat{y} = \theta_0 x + \sin(\theta_1 x)$
* **Characteristics:** Combines a linear trend with a periodic oscillation. Because of the sine wave frequency parameter, this introduces a non-convex loss surface with multiple local minima, requiring careful learning rate and initialization choices.

### 3. Polynomial Model (`poly_model.py`)
* **Mathematical Form:** Higher-order feature expansions (e.g., cubic fields like $x^2$ and $x^3$).
* **Characteristics:** Demonstrates how feature engineering allows a standard linear setup to map complex non-linear geometric curves, while highlighting the engineering trade-offs between underfitting and high-variance overfitting.

---

## Requirements & Usage

### Prerequisites
* Python 3.x
* NumPy
* Pandas
* Matplotlib or Plotly
* Seaborn (for statistical data visualization)

### Running a Model
Execute any of the model scripts directly to process the data, run the optimization loop, and view the visualization plots:
```bash
python linear_model.py
python sin_model.py
python poly_model.py
