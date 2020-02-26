# MNIST Digit Recognition

<p align="center">
<img src="https://raw.githubusercontent.com/master-coro/mnist-digit-recognition/master/res/MNIST.png">
</p>

Based on handwritten digits - which a sampled is displayed hereinabove -  stored in the MNIST database (/res/...) and parsed by the great and useful toolbox [mnist](https://github.com/datapythonista/mnist) written by [Marc Garcia](https://github.com/datapythonista). This program uses feedforward and backpropagation in a 3-layers Neural Network seen below to learn how to recognize the former mentionned digits :

<p align="center">
<img src="https://raw.githubusercontent.com/master-coro/mnist-digit-recognition/master/res/iHDtO.png">
</p>

## Done

* Feedforward
* Backpropagation
* Gradient Descent
* Fit
* Predict
* Accuracy Test

## To Do

* Customizable number of layers
* Cool terminal-based UI
* Continuous Integration (GitHub Actions, Pytest)
* Use none-MNIST handwritten digit as inputs
* Any more ideas ? Open an Issue !

## Loss and Accuracy

For a batch size of one, five epochs and a learning rate of 0.01, we get these graphs :

<p align="center">
<img src="https://raw.githubusercontent.com/master-coro/mnist-digit-recognition/master/res/lossfunction.png">
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/master-coro/mnist-digit-recognition/master/res/accuracyfunction.png">
</p>

## Setup

To setup the project on your local machine:

1. Click on `Fork`.
2. Go to your fork and `clone` the project to your local machine.
3. `git clone https://github.com/master-coro/mnist-digit-recognition`

## Run

To run the project:
1. Cd into the root of the project `cd path/mnist-digit-recognition`
2. Run the main script with necessary args : `python src/main.py`

## Contribute

To contribute to the project:

1. Choose any open issue from [here](https://github.com/master-coro/mnist-digit-recognition/issues). 
2. Comment on the issue: `Can I work on this?` and get assigned.
3. Make changes to your fork and send a PR.

To create a PR:

Follow the given link to make a successful and valid PR: https://help.github.com/articles/creating-a-pull-request/

To send a PR, follow these rules carefully,**otherwise your PR will be closed**:

1. Make PR title in this format: `Fixes #IssueNo : Name of Issue`

For any doubts related to the issues, i.e., to understand the issue better etc, comment down your queries on the respective 
