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
# Build and Run (recommended)
g++ -g -std=c++17 src/_mainmodel.cpp -o bin/_mainmodel.exe; .\bin\_mainmodel.exe
```

## Train (uncomment in main()):

```cpp
buildTree(trainData, maxDepth, maxLeafNodes, ..., minSamplesSplit, minSamplesLeaf);
```

## Predict:

```cpp
loadDecisionTree("treesave.txt");
```

## Comparison with sklearn

### Running the Comparison

Compare the C++ decision tree with scikit-learn's implementation:

```bash
# First, build and run the C++ tree to generate predictions
g++ -g -std=c++17 src/_mainmodel.cpp -o bin/_mainmodel.exe
.\bin\_mainmodel.exe

# Then run the comparison script
python src/compare_trees_report.py
```

### What the Comparison Shows

The script (`src/compare_trees_report.py`) generates a detailed report (`comparison_report.txt`) that includes:

- **sklearn Training Accuracy**: Default sklearn tree (usually 100% - overfits)
- **Tree Properties**: Depth, number of leaves, feature importance
- **Prediction Comparison**: 
  - Side-by-side predictions on test set
  - Matching rate percentage
  - List of differences
