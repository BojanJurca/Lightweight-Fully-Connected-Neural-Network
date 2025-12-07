#include <array.hpp>
#include <iostream.hpp>


// ----- the neural network -----

    #include "neuralNetwork.hpp"
    
    //                   .--- the number inputs
    //                   |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     .--- the number of neurons in the first layer
    //                   |      |     |                                      .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     |                                      |     .--- the number of neurons in the output layer = the number of outputs
    //                   |      |     |                                      |     |
    neuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> neuralNetwork;
    // at this point neuralNetwork is initialized with random weights and biases and it is ready for training
    // - you can either start training it and export the trained model when fiished
    // - or you can load already trained model that is cappable of making usable outputs 


void setup () {

    cinit ();

    // load pre-trained model
    neuralNetwork = {0x1.099fp+0f,0x1.072ed4p+0f,-0x1.14b4fcp+1f,-0x1.0ef47ep+1f,-0x1.71b4bp+0f,0x1.60bf78p-1f,-0x1.1519e4p+1f,-0x1.fdc5b8p+0f,-0x1.8f5b94p-1f};

    cout << "the neural network is trained and ready\n\n";
    cout << fixed << setprecision (6);

    // forwardPass returns an array of output values.  
    // In this case the array has dimension 1, so the only output value  
    // is located at index 0.

    cout << "0 xor 0 = " << neuralNetwork.forwardPass ({0, 0}) [0] << endl;
    cout << "0 xor 1 = " << neuralNetwork.forwardPass ({0, 1}) [0] << endl;
    cout << "1 xor 0 = " << neuralNetwork.forwardPass ({1, 0}) [0] << endl;
    cout << "1 xor 1 = " << neuralNetwork.forwardPass ({1, 1}) [0] << endl;
}


void loop () {

}
