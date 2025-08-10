# Assignment 1: Dimensionality Reduction

## Objectives

By completing this assignment, you will:

- Understand and implement some commonly used dimensionality reduction techniques
- Design and justify appropriate evaluation metrics.
- Analyze and interpret the impact of dimensionality reduction on a given dataset.
- Learn to use AI tools responsibly in your coding and reporting process.
---

## Task Overview

1. **Implementation:**

   ⚠️ **IMPORTANT**: You can use AI tools to assist with coding (e.g., explaining, commenting, debugging), 
      but you must write the core logic yourself. All your AI chat history must be submitted with your report.

   - Load the provided dataset (16000 x 128).
   - Apply **row-wise** z-normalization on the dataset.
   - Compute the distance matrix of the normalized dataset using Euclidean distance.
   - Implement the following dimensionality reduction techniques using Python (you can use `numpy` for numerical 
   operations, but you **CANNOT** directly call functions from `scikit-learn` or similar libraries):
     - Principal Component Analysis (PCA)
     - Wavelet Transform using Haar matrix
   - Transform the normalized dataset using these techniques.
     - For PCA, please keep the **best** (NOT the first or last) 4 principal components.
     - For Wavelet transform, use the first 4 coefficients.
   - Compute the Euclidean distance matrix the reduced datasets.
   - Benchmark each technique's performance:
     1. Time: Measure the time taken to apply each technique.
     2. Relationship consistency: For each pair of cells in the distance matrix, check if the distance relationships (greater or 
        less) hold in the reduced matrices compared to the original matrix.

        *Example*: If the distance between points A and B is greater than the distance between A and C in the original matrix,
        check if this relationship holds in the reduced matrices as well. We want to see how often these relationships are preserved.
        
        If the computation is too time-consuming, consider using a random sample of pairs instead of all pairs.

     3. With the help of AI tools, find another metric that can be used to compare the dimensionality reduction techniques.

2. **Report:**

   Write a report yourself (AI tool **NOT** allowed here) that includes:
   - An introduction to each technique and implementation summary
   - Justification of your evaluation metrics
   - Results, discussion of tradeoffs, and key insights
   - AI usage reflection
   
3. **AI Usage Reflection:**
   
   Briefly describe your thought on how you used AI tools. What worked well? What didn't? How did it help you learn or improve your work? 
   Please also append your chat history with AI tools at the end of your report.

---

## Flavor Options

We prepared two flavors of this assignment to cater to different preferences. 
You have access to both and you can choose either version based on your preference.

- **Version 1:** In this version, you will be given a Jupyter Notebook with faulty implementation, as well as a chat history with
  AI tools that generated the code. Your task is to analyze the code, identify and fix the issues, and complete the analysis.
- **Version 2:** You will get a Jupyter Notebook template with the predefined structure and some 
  starter code. Your task is to fill in the missing parts, implement the techniques, and complete the analysis.

**Note:** 
- The two versions are designed to be equivalent in terms of learning outcomes, so choose the one you find more engaging.
- Do NOT change the function name or function parameters in the provided code. 
  The evaluation code will call your functions with specific parameters, and changing them will result in an error.

---

## Get Started
1. Clone this assignment repository from GitHub Classroom.
2. Choose the version you prefer (Version 1 or Version 2), write your code in the corresponding Jupyter Notebook (`./version_X/assignment_1.ipynb`), and complete the analysis.
3. You can run the evaluation code at the end of the notebook to verify your implementation correctness.
---

## Coding Guidelines
- Good readability:
  - Use functions or classes to encapsulate logic, avoid boilerplate code.
  - Use meaningful variable and function names.
  - Add comments to explain complex logic.
  - Use docstrings for functions to describe their purpose, parameters, and return values.
  - For additional information about Python coding style, you can read the PEP8 document (https://www.python.org/dev/peps/pep-0008/).
- Use appropriate libraries for numerical operations and plotting (e.g., `numpy`, `matplotlib`).
- Ensure reproducibility by setting random seeds where applicable.

---

## AI Usage Policy

**AI tools are allowed, but only if used responsibly:**

| Allowed                          | Not Allowed                                 |
|----------------------------------|---------------------------------------------|
| Debugging help                   | Copy-pasting entire solutions               |
| Summarizing, commenting          | Using AI to generate full report or metrics |
| Improving report clarity/grammar | Submitting unedited AI output as your work  |


---

## Submission Checklist

We will use GitHub Classroom to collect your code. In addition, please submit the PDF report on Canvas.

- [ ] Code: 
  - [ ] Commit and push the completed notebook `assignment_1.ipynb` to the assignment repository.
- [ ] PDF Report
  - [ ] Main content
  - [ ] AI Usage Reflection section
  - [ ] Chat history with AI tools

---

## Suggested Libraries

- `numpy` for numerical operations (https://numpy.org/)
- `matplotlib` for plotting (https://matplotlib.org/)
- `time` for elapsed time measurement
- `random` for random sampling (if needed)
