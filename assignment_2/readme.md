# Assignment 2: Linear Classification & Impact of Dimensionality Reduction

## Objectives

By completing this assignment, you will:

- Implement the classic Perceptron learning algorithm from scratch.
- Implement and apply cross-validation.
- Analyze how projecting high-dimensional data to a lower-dimensional subspace impacts linear separability and convergence speed.
- Practice careful experimental design, reproducibility, and responsible AI tool usage.
---

## Task Overview

You will work with a provided dataset (`data/assignment_2_data.txt`). The (implicit) class labels for this assignment are:
- Rows 0–7999  → Class 0
- Rows 8000–15999 → Class 1

### 0. Data Preparation & Shuffling
- Load the 16000 x 128 dataset.
- Construct the binary label vector y using the rule above.
- Shuffle the rows (features AND labels together) before performing any cross-validation split. Use a fixed random seed for reproducibility.

### 1. Perceptron on Original 128-D Space
Implement the standard (binary) Perceptron following the update rule in:
https://en.wikipedia.org/wiki/Perceptron

Requirements:
- No library model trainers (e.g., no scikit-learn classifiers). Only use numpy for numeric ops.
- Use 5-fold cross-validation (since the dataset is balanced, simple partition after shuffling is fine).
- For each fold, report:
  1. Fold accuracy (% correct predictions on the validation fold)
  2. Number of training iterations executed (until convergence or stopping criterion is met)
- Convergence / stopping:
  - Use a misclassification threshold $\gamma$ (gamma). Suggested $\gamma$ = 0.1 (i.e., stop if validation or training misclassification rate ≤ γ — clarify which you choose and be consistent). OR stop earlier if zero classification errors occur.
  - Also enforce a hard cap on iterations (suggested `max_iter` = 1000) to avoid infinite loops on non-linearly separable subsets.
- Clearly document: learning rate, initialization strategy (zeros vs small random), stopping conditions, etc.

### 2. Perceptron After Dimensionality Reduction (Haar Wavelet Transform)
- You will be provided with a Haar Wavelet transform function.
- Transform the (shuffled) data and keep only the first 4 features.
- Repeat the same 5-fold Perceptron training & reporting procedure on the 4-D representation.
- Report per-fold accuracy and iteration counts.

### 3. Comparative Analysis (to include in your report)
- Compare: Accuracy distribution (fold-wise) in 128-D vs 4-D.
- Compare: Iterations to stopping — does reduced dimensionality help convergence? Why or why not?
- Discuss: Influence of dimensionality on linear separability, potential information loss, and stability across folds.
- OPTIONAL (extra depth): Track margin evolution, or plot cumulative mistakes per iteration for one representative fold in both settings.

### 4. AI Usage Reflection
- Reflect on how you used AI tools during this assignment.
- Include the chat history with the AI tool in your report.

### Rules & Constraints
- You must not call any library function that directly trains a perceptron. The core logic must be implemented by yourself.
- Arithmetic, vectorized numpy operations, and basic linear algebra are allowed.
- Start with a smaller subset to debug before scaling to all 16000.

### Notes
- Suggested gamma ($\gamma$) = 0.1 — justify if you choose differently.
- Suggested maximum iterations = 1000.
- Define clearly whether $\gamma$ applies to training or validation misclassification for early stopping (training is typical). Be consistent across both experiments.
- Use the same random seed for:
  - Shuffling
  - Any random initialization of weights (if used)

### Recommended Output Tables
1. Perceptron (128-D): Fold | Accuracy (%) | Iterations
2. Perceptron (4-D Haar): Fold | Accuracy (%) | Iterations
3. Summary: Mean ± Std for both accuracy and iterations in each setting.

---

## Flavor Options
Like Assignment 1, you will have the option of two different "flavors" for this assignment:
- **Version 1**: Debugging a faulty Perceptron implementation with provided AI chat history.
- **Version 2**: Implementing the Perceptron from scratch.

---

## AI Usage Policy

AI tools are allowed, but only if used responsibly:

| Allowed                          | Not Allowed                                   |
|----------------------------------|-----------------------------------------------|
| Debugging help                   | Delegating full Perceptron implementation     |
| Summarizing, commenting          | Auto-generating the full analysis section     |
| Brainstorming evaluation ideas   | Letting AI pick hyperparameters without review|
| Improving report clarity/grammar | Submitting unmodified AI-produced report text |

You must still write and understand the core training logic.

---

## Submission Checklist
We will use GitHub Classroom to collect your code. Also submit the PDF report on Canvas.

- [ ] Code:
  - [ ] Completed `version_X/assignment_2.ipynb` with implementation & results.
- [ ] Report (PDF):
  - [ ] Methodology & algorithm description
  - [ ] Experimental setup (CV, gamma, iteration cap)
  - [ ] Results tables & plots
  - [ ] Comparative analysis (128-D vs 4-D)
  - [ ] AI Usage Reflection + Chat history

---

## Suggested Libraries
- numpy for numerical operations
- time for timing loops / iteration throughput
- matplotlib (optional) for mistake vs iteration plots
- random (optional) for shuffling (though numpy preferred)

(Do NOT use `scikit-learn` or other ML training utilities for the Perceptron itself.)


