# MNIST Neural Network in Julia

This project implements a convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset. The model is built using the [Flux.jl](https://fluxml.ai/) library and evaluates performance with metrics such as accuracy, recall, precision, specificity, and F1 score.

## Requirements

Ensure you have Julia installed. The following packages are required:

- **Flux.jl**: For building and training the neural network.
- **MLDatasets.jl**: For loading the MNIST dataset.
- **Statistics.jl**: For computing metrics.

Install the packages by running:
```julia
using Pkg
Pkg.add(["Flux", "MLDatasets", "Statistics"])
```

## Running the Project

1. Clone the repository or download the `mnist_neural_network.jl` file.
2. Open the file in a Julia environment.
3. Run the script to train the model and evaluate its performance:
    ```julia
    include("mnist_neural_network.jl")
    ```
4. View the accuracy and metrics for each class in the output.

## Features

- Convolutional layers for feature extraction.
- Metrics for detailed evaluation of each class.
- Adjustable training parameters (epochs, batch size).
- Easy-to-use implementation for MNIST classification.
- Performance evaluation on test data.

---

# MNIST Random Forest in Julia

This project implements a Random Forest classifier for classifying handwritten digits from the MNIST dataset. The model is built using the [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) library and evaluates performance with metrics such as accuracy, recall, precision, specificity, and F1 score.

## Requirements

Ensure you have Julia installed. The following packages are required:

- **DecisionTree.jl**: For building and training the Random Forest model.
- **MLDatasets.jl**: For loading the MNIST dataset.
- **Statistics.jl**: For computing metrics.

Install the packages by running:
```julia
using Pkg
Pkg.add(["DecisionTree", "MLDatasets", "Statistics"])
```

## Running the Project

1. Clone the repository or download the `mnist_random_forrest.jl` file.
2. Open the file in a Julia environment.
3. Run the script to train the model and evaluate its performance:
    ```julia
    include("mnist_random_forrest.jl")
    ```
4. View the accuracy and metrics for each class in the output.

## Features

- Random Forest classifier with adjustable parameters (number of trees, max depth, partial samppling).
- Metrics for detailed evaluation of each class.
- Easy-to-use implementation for MNIST classification.
- Performance evaluation on test data.
- Efficient training and prediction for large datasets.
- Supports hyperparameter tuning for optimal performance.
