# neuralNetwork.hpp – Lightweight Neural Network Library for Arduino 🤖⚡

A minimalistic implementation of a fully connected neural network designed for **Arduino controllers** (including AVR) and other **Arduino‑compatible platforms**.  
The library uses **no heap memory**, making it ideal for devices with very limited resources.

---

## ✨ Features

- ✅ Zero dynamic memory allocation (`malloc`/`new` free)  
- ✅ Compatible with **AVR**, **ESP32**, standard C++ and more  
- ✅ Train models on a computer, then deploy to Arduino  
- ✅ Activation functions: **Sigmoid**, **ReLU**, **Tanh**, **FastTanh**  
- ✅ Built‑in **softmax normalization** for output layers  
- ✅ Flexible topology via **C++ variadic templates**  

---

## 📦 Dependencies

The library relies on [LightweightSTL](https://github.com/) to ensure compatibility with standard C++.

---

## 🚀 Quick Start

### Installation
1. Add `neuralNetwork.hpp` to your project.  
2. Install **LightweightSTL**.  

### Example

```cpp
#include "neuralnetwork.hpp"

// Define a network: 8 inputs → hidden layer with 16 neurons (ReLU) → output layer with 2 neurons (Sigmoid)
neuralNetworkLayer_t<8, ReLU, 16, Sigmoid, 2> neuralNetwork;

void setup() {
    cinit(); // instead of Serial.begin()

    // Import a pre‑trained model
    neuralNetwork = {0x1.099fp+0f, -0x1.14b4fcp+1f, ...};

    // Run inference
    auto probability = softmax(neuralNetwork.forwardPass({18, 20, 7, 2, 1}));

    cout << "Probabilities: ( ";
    for (auto p : probability) cout << p << " ";
    cout << ")\n";
}




🧠 Training
Training can be performed on a PC (recommended) or directly on Arduino.

Supports backpropagation with Xavier and He initialization.

Export trained models as C++ initializer lists for easy deployment.

Example training loop:



#define epoch 1000
for (int i = 0; i < epoch; i++) {
    loss += neuralNetwork.backwardPropagation({1,2,6,18,20}, {1,0,0});
    loss += neuralNetwork.backwardPropagation({1,2,25,3,1}, {0,1,0});
    loss += neuralNetwork.backwardPropagation({19,10,3,2,1}, {0,0,1});
}
cout << hexfloat << neuralNetwork << endl;




📐 Theory Overview
Forward pass: compute outputs from inputs

Backward propagation: update weights and biases using gradient descent

Initialization methods: Xavier (for Sigmoid/Tanh) and He (for ReLU)

Random values generated via Box‑Muller transform



📊 Typical Use Cases
Classifying sensor data or input patterns

Running lightweight ML models on microcontrollers

Experimenting with neural networks without large frameworks


⚠️ Notes
Training on Arduino is limited – prefer training on a PC.

Beware of overfitting – more training does not always mean better accuracy.

Bias values are typically initialized to 0.

📚 References
3Blue1Brown – Backpropagation calculus

Documentation for LightweightSTL



