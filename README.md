# ML_COURSEBARI

The `ClassTrainingData5SigFigs.txt` file contains 30k examples, each having the first column with the true label and 23 inputs.
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
$ python Classifier_FraBru.py
```

The command will produce `.png` files:
- ROC curves `ROC_results.png`
- Loss function vs Trainsize `numbertrainingsamples_loss_mse.png`
- MLP score vs Trainsize `numbertrainingsamples_MLP_score.png`

Examples of the plot produced are stored in the `Plots_NNtraining\` directory for the usage of a Multi Layer Perceptron (MLP) and a Boosted Decision Tree (BDT) implemented by using scikit-learn package.
