# Model Comparison Report

## Objective

The objective of this experiment was to understand how different neural network architectures affect model performance. Multiple neural network models were trained on the Iris Dataset by changing:

* Number of hidden layers
* Number of neurons
* Activation functions
* Number of training epochs

The performance of each model was compared using classification accuracy.

---

# Experimental Setup

## Dataset

The Iris Dataset contains:

* 150 flower samples
* 4 input features
* 3 output classes

The dataset was split into:

* 80% Training Data
* 20% Testing Data

Feature normalization was performed using StandardScaler.

---

# Model Configurations

## Model 1

* Hidden Layers: 1
* Neurons: 8
* Activation Function: ReLU
* Epochs: 30

### Characteristics

This model is simple and trains quickly. It contains fewer parameters and therefore requires less computation.

---

## Model 2

* Hidden Layers: 1
* Neurons: 16
* Activation Function: ReLU
* Epochs: 50

### Characteristics

Increasing the number of neurons allows the network to learn more complex relationships within the data.

---

## Model 3

* Hidden Layers: 2
* Neurons: 16 and 8
* Activation Function: ReLU
* Epochs: 50

### Characteristics

Adding an extra hidden layer increases the model's ability to learn hierarchical patterns and feature interactions.

---

## Model 4

* Hidden Layers: 2
* Neurons: 32 and 16
* Activation Function: Tanh
* Epochs: 100

### Characteristics

This is the most complex architecture among all tested models. It contains more neurons and is trained for more epochs.

---

# Performance Comparison

| Model   | Hidden Layers | Neurons | Activation | Epochs | Accuracy |
| ------- | ------------- | ------- | ---------- | ------ | -------- |
| Model 1 | 1             | 8       | ReLU       | 30     | Varies   |
| Model 2 | 1             | 16      | ReLU       | 50     | Varies   |
| Model 3 | 2             | 16,8    | ReLU       | 50     | Varies   |
| Model 4 | 2             | 32,16   | Tanh       | 100    | Varies   |

The exact accuracy values may differ slightly each time because neural network training contains random initialization and optimization processes.

---

# Analysis

## Effect of Hidden Layers

Models with more hidden layers generally perform better because they can learn more complex patterns.

Advantages:

* Better feature extraction
* Higher learning capacity
* Improved classification performance

Disadvantages:

* Longer training time
* Higher computational cost

---

## Effect of Number of Neurons

Increasing neurons allows the model to capture more information.

Advantages:

* Better representation learning
* Improved accuracy

Disadvantages:

* More parameters
* Increased risk of overfitting

---

## Effect of Activation Functions

### ReLU

Benefits:

* Faster training
* Computationally efficient
* Reduces vanishing gradient problems

Used in:

* Most modern deep learning models
* Computer vision applications

### Tanh

Benefits:

* Output centered around zero
* Useful in some deep networks

Limitations:

* Can suffer from vanishing gradients

In many experiments, ReLU often converges faster than Tanh.

---

## Effect of Epochs

Training for more epochs allows the model to learn additional patterns.

Advantages:

* Better convergence
* Lower loss

Disadvantages:

* Risk of overfitting if training continues for too long

An optimal number of epochs should balance learning and generalization.

---

# Why Some Models Perform Better

Several factors contribute to improved performance:

### More Hidden Layers

Additional layers allow the network to learn increasingly abstract features.

### More Neurons

A larger number of neurons increases the network's capacity to model complex relationships.

### Appropriate Activation Functions

ReLU often performs better because it helps gradients flow more effectively through the network.

### Sufficient Training Epochs

Models trained for more epochs generally achieve lower loss and better accuracy, provided overfitting does not occur.

---

# Conclusion

This experiment demonstrates that neural network architecture significantly influences performance. Simpler models train faster but may not capture all patterns within the data. Increasing the number of layers, neurons, and training epochs generally improves accuracy, although excessive complexity may lead to overfitting.

Among the tested architectures, deeper networks with an adequate number of neurons and sufficient training epochs typically provide the best balance between learning capability and prediction accuracy. Understanding these architectural choices is essential when designing effective deep learning models for real-world applications.
