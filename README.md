<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#ML_COURSEBARI">The exercise</a>
      <ul>
        <li><a href="##Authors">Authors</a></li>
        <li><a href="##Requirements">Requirements</a></li>
        <li><a href="##Instructions">Instructions</a></li>
        <li><a href="##Your-own-Neural-Network-from-scratch">Your own Neural Network from scratch</a></li>
      </ul>
    </li>
  </ol>
</details>

# ML_COURSEBARI

The `NormDat_test.txt` and `NormDat_train.txt` files contain 30.000 randomized and normalized examples, each having the first column with the true label and 23 particle physics features.
This is used by a ML class in Bari for testing different Machine Learning algorithms and their performance

Darwin MacBook-Pro.local 21.4.0 Darwin Kernel Version 21.4.0: Fri Mar 18 00:47:26 PDT 2022; root:xnu-8020.101.4~15/RELEASE_ARM64_T8101 arm64

## Authors

- [Brunella D'Anzi](https://github.com/bdanzi) (University and INFN Bari)
- [Francesco Sivo](https://github.com/FrancescoSivo) (University and INFN Bari)

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

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/bdanzi/ML_COURSEBARI.svg?style=for-the-badge
[contributors-url]: https://github.com/bdanzi/ML_COURSEBARI/contributors

[forks-shield]: https://img.shields.io/github/forks/bdanzi/ML_COURSEBARI.svg?style=for-the-badge
[forks-url]: https://github.com/bdanzi/ML_COURSEBARI/network/members

[stars-shield]: https://img.shields.io/github/stars/bdanzi/ML_COURSEBARI.svg?style=for-the-badge
[stars-url]: https://github.com/bdanzi/ML_COURSEBARI/stargazers

[issues-shield]: https://img.shields.io/github/issues/bdanzi/ML_COURSEBARI.svg?style=for-the-badge
[issues-url]: https://github.com/bdanzi/ML_COURSEBARI/issues

[license-shield]: https://img.shields.io/github/license/bdanzi/ML_COURSEBARI.svg?style=for-the-badge
[license-url]: https://github.com/bdanzi/ML_COURSEBARI/blob/main/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/brunella-d-anzi

