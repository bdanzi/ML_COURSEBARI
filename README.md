# ML_COURSEBARI

The `ClassTrainingData5SigFigs.txt` file contains 30k examples, each having the first column with the true label and 23 inputs.
This is used by a ML class in Bari for testing different Machine Learning algorithms and their performance

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
