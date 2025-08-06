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

   - Implement the following dimensionality reduction techniques using Python without using any high-level libraries:
     - Principal Component Analysis (PCA)
     - Wavelet Transform using Haar matrix
   - Transform a given dataset (16000 x 128) using these techniques.
   - Compute the Euclidean distance matrix for the original and reduced datasets.
   - Benchmark each technique's performance:
     1. Time: Measure the time taken to apply each technique.
     2. Relationship consistency: For each pair of cells in the distance matrix, check if the relationships (greater or 
        less) hold in the reduced matrices compared to the original matrix.

        *Example*: If the distance between points A and B is greater than the distance between A and C in the original matrix,
        check if this relationship holds in the reduced matrices as well. We want to see how often these relationships are preserved.
        
        If the computation is too time-consuming, consider using a random sample of pairs instead of all pairs.

     3. With the help of AI tools, find another metric that can be used to compare the dimensionality reduction techniques.

2. **Two versions:**
  
    - **Version 1:** In this version, you will be given a Python script with faulty implementation, as well as a chat history with
      AI tools that generated the code. Your task is to analyze the code, identify and fix the issues, and complete the analysis.
    - **Version 2:** You will get a Jupyter Notebook template with the predefined structure and some 
      starter code. Your task is to fill in the missing parts, implement the techniques, and complete the analysis.
    - You can choose either version based on your preference.
    - **Note:** The two versions are designed to be equivalent in terms of learning outcomes, so choose the one you find more engaging.
    - 
3. **Report:**

   Write a report yourself (AI NOT allowed here) that includes:
   - An introduction to each technique and implementation summary
   - Justification of your evaluation metrics
   - Results, discussion of tradeoffs, and key insights
   - AI usage reflection
   
4. **AI Usage Reflection:**
   
   Briefly describe your thought on how you used AI tools. What worked well? What didn't? How did it help you learn or improve your work? 
   Please also append your chat history with AI tools at the end of your report.

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

- [ ] Implementation (Python script or Jupyter Notebook depending on flavor)
  - [ ] Flavor 1: Python script or Jupyter Notebook with fixed issues and completed analysis
  - [ ] Flavor 2: Jupyter Notebook (an `.ipynb` file) with clear code, comments, and plots
- [ ] PDF Report
  - [ ] Main content
  - [ ] AI Usage Reflection section
  - [ ] Chat history with AI tools

---

## Suggested Libraries

- `numpy` for numerical operations
- `matplotlib` for plotting
