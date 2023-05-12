# Digit Classification Neural Network

This repository contains a digit classification neural network implemented in Python using only the NumPy library. The digit classification neural network is designed to recognize and classify handwritten digits. It is implemented using a three-layer feedforward neural network architecture with fully connected layers. The network is trained using gradient descent optimization and backpropagation for parameter updates.

By leveraging the power of deep learning techniques, the network is capable of learning patterns and features present in handwritten digits, enabling accurate and automated recognition. The goal of this project is to develop a robust and efficient neural network model that can accurately classify digits from 0 to 9.

The neural network's performance has been optimized using various techniques. Momentum and RMS propagation, Adam optimization that combines both of them for faster and more stable convergence. Gradient clipping prevents exploding gradients. Additionally, the network has been expanded with more layers and units.

The neural network is trained on the MNIST dataset. This dataset serves as an ideal benchmark for evaluating the performance of digit classification algorithms. Through this repository, you can explore the implementation details, train the neural network on the MNIST dataset, evaluate its performance, and visualize its predictions on sample images.

## Table of Contents
- [Features](#features)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)


## Features

-   **Neural Network Architecture:** The digit classification neural network is implemented using a three-layer feedforward neural network architecture with fully connected layers. The input layer takes the pixel values of the handwritten digits as inputs. The hidden layer employs the Rectified Linear Unit (ReLU) activation function, which introduces non-linearity and helps the network learn complex representations. The ReLU activation function selectively activates neurons based on their positive input values, allowing the network to capture and amplify relevant features. The output layer utilizes the Softmax activation function, which transforms the outputs into probabilities, representing the likelihood of each digit class. The Softmax function ensures that the sum of the probabilities across all output neurons is equal to 1, enabling the network to make confident predictions based on the highest probability. This architecture, combined with ReLU and Softmax activation functions, enables the network to effectively learn and classify the handwritten digits.
    
-   **Optimizations:** To enhance the performance of the neural network, several optimization techniques have been incorporated:
    
    -   **Momentum:** Momentum optimization has been implemented to accelerate the convergence of the training process. This helps to overcome local minima and speeds up the learning process.
        
    -   **RMS Propagation:** RMS propagation, or Root Mean Square Propagation, is employed to adjust the learning rate adaptively for each weight based on the magnitude of recent gradients. This technique speeds up convergence and prevents oscillations during training.
        
    -   **Adam Optimization:** The neural network utilizes the Adam optimization algorithm, which combines the concepts of momentum and RMS propagation. Adam optimization adapts the learning rate for each weight parameter individually, resulting in faster and more stable convergence.
        
    -   **Gradient Clipping:** Gradient clipping has been applied to prevent exploding gradients during training. It limits the magnitude of gradients to a predefined threshold, ensuring stable learning and preventing numerical instability.
        
    -   **More Layers and Units:** The neural network architecture has been expanded with additional layers and units. This increase in model capacity allows the network to capture more complex patterns and features in the input data.
        
-   **Training and Evaluation:** The neural network is trained using gradient descent optimization and backpropagation for parameter updates. The MNIST dataset, consisting of 60,000 training images and 10,000 test images of handwritten digits, is used to train and evaluate the performance of the neural network. The goal is to achieve high accuracy in classifying the digits from 0 to 9.
    
-   **Accuracy Measurement and Visualization:** After training, the neural network's accuracy on the MNIST test set is measured. The performance of the network can be evaluated and visualized through accuracy measurement and the visualization of training progress.

## Results

The trained neural network achieves an impressive accuracy of 98% on the MNIST test set. This indicates its exceptional ability to accurately classify handwritten digits. The dataset consists of grayscale images with a resolution of 25x25 pixels. The high accuracy showcases the effectiveness of the implemented neural network architecture and the success of the training process.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Any contributions that align with the goals of the project are highly appreciated.

## License

This project is licensed under the MIT License