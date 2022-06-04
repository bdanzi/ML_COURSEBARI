# ML_COURSEBARI

The `NormDat_test.txt` and `NormDat_train.txt` files contain 30 000 randomized and normalized examples, each having the first column with the true label and 23 particle physics features.
This is used by a ML class in Bari for testing different Machine Learning algorithms and their performance

Darwin MacBook-Pro.local 21.4.0 Darwin Kernel Version 21.4.0: Fri Mar 18 00:47:26 PDT 2022; root:xnu-8020.101.4~15/RELEASE_ARM64_T8101 arm64
## Requirements

```bash
$ pip install numpy
$ pip install sklearn
$ pip install matplotlib
$ pip install pandas
```

## Instructions

In the `ML_COURSEBARI\` directory of this repository:

```bash
$ python ml_exercise.py
```

The command will produce `.png` files:
- ROC curves `ROC_results.png`
- Loss function vs Trainsize `numbertrainingsamples_loss_mse.png`
- MLP score vs Trainsize `numbertrainingsamples_MLP_score.png`

Examples of the plot produced are stored in the `Plots_NNtraining\` directory for the usage of a Multi Layer Perceptron (MLP) and a Boosted Decision Tree (BDT) implemented by using scikit-learn package.

# Your own Neural Network from scratch

```bash
$ python NN_backpropagation_bdanzi.py
```

This python code has been written to implement the backpropagation process in Neural Network training not using already available open-source packages.
The command will produce the `.png` file:
- Loss function vs Epochs
This code represents a home-made (no usage of numpy, scikit-learn,keras) Neural Network having one hidden layer.
