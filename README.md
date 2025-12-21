# Decision Tree from Scratch (C++)

## Overview

C++ implementation of a supervised Decision Tree classifier from scratch, focusing on core machine learning concepts without using external ML libraries.

## Key Features

- Decision Tree using entropy & information gain
- Numeric feature splitting
- Pre-pruning to reduce overfitting:
  - Max depth
  - Max leaf nodes
  - Min samples per split / leaf
- Tree save & load
- Prediction on unseen data

## Dataset

- **Train:** label + 4 integer features (L, R, B)
- **Test:** 4 integer features

## How It Works

- Select best split using information gain
- Recursively build tree
- Stop growth based on pruning constraints
- Predict by traversing tree from root to leaf

## Run

```bash
# Build (recommended)
g++ -std=c++17 -g _mainmodel.cpp -o _mainmodel.exe

# Or use MSYS2 g++ explicit path:
# D:\msys2\ucrt64\bin\g++.exe -std=c++17 -g _mainmodel.cpp -o _mainmodel.exe
```

## Train (uncomment in main()):

```cpp
buildTree(trainData, maxDepth, maxLeafNodes, ..., minSamplesSplit, minSamplesLeaf);
```

## Predict:

```cpp
loadDecisionTree("treesave.txt");
```

## What I Learned

- Internal mechanics of decision trees
- Overfitting control via pre-pruning
- Recursive tree structures in C++

## Tech

C++ · Supervised Learning · Decision Trees

