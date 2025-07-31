# 🧪 Neural Network on Tabular Data – Activation & Optimizer Comparison (PyTorch)

This project explores how different neural network configurations affect performance on a 2D tabular dataset. Using PyTorch, I trained a feedforward network under 8 combinations of activation functions and optimizers.

## 📌 Project Overview

- Dataset: Custom 2D feature classification dataset (2 classes)
- Model: Fully connected NN with 5 layers, BatchNorm, and ReLU/LeakyReLU
- Objective: Compare the effects of:
  - Optimizer: Adam vs SGD
  - Activation: ReLU vs LeakyReLU
  - Learning rate: 0.001 vs 0.01

## ⚙️ Experiments

A total of **8 configurations** were tested:
- Each combination of 2 activations × 2 optimizers × 2 learning rates
- For each run, training and validation accuracy were recorded
- Visualizations include decision boundaries and ROC curves

## 📊 Results

- Highest AUC achieved with **ReLU + Adam + LR=0.001**
- Decision boundaries were more stable with Adam
- ROC curves show clearer separation with ReLU over LeakyReLU

_See plots folder or output graphs for comparisons._

## 🧠 Key Learnings

- Optimizers and activations can significantly impact small tabular datasets
- Adam handled the optimization more robustly across configurations
- Visualizing ROC curves and decision boundaries helped identify overfitting/underfitting

