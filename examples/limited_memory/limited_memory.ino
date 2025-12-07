#include <array.hpp>
#include <iostream.hpp>


// ----- the neural network -----

    #include "neuralNetwork.hpp"
    
    //                           .--- the number inputs
    //                           |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                           |      |     .--- the number of neurons in the first layer
    //                           |      |     |                                      .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                           |      |     |                                      |     .--- the number of neurons in the output layer = the number of outputs
    //                           |      |     |                                      |     |
    typedef neuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> neuralNetwork_t;

    // Define the neural network as a union with a float array.  
    // This small trick ensures that only one copy of the initialization data is stored in RAM.  
    // Note that the neural network is initialized directly from the pre-trained model,  
    // without calling the class constructor, which would normally initialize weights and  
    // biases with random values to prepare the network for training.  
    // If you intend to train the network, you must explicitly call n.n.randomize()  
    // before starting the training process.

    static union {
        float model [sizeof (neuralNetwork_t) / sizeof (float)];
        neuralNetwork_t n;
    } n = {0x1.099fp+0f,0x1.072ed4p+0f,-0x1.14b4fcp+1f,-0x1.0ef47ep+1f,-0x1.71b4bp+0f,0x1.60bf78p-1f,-0x1.1519e4p+1f,-0x1.fdc5b8p+0f,-0x1.8f5b94p-1f};


void setup () {

    cinit ();

    cout << "the neural network is trained and ready\n\n";
    cout << fixed << setprecision (6);

    // forwardPass returns an array of output values.  
    // In this case the array has dimension 1, so the only output value  
    // is located at index 0.    

    cout << "0 xor 0 = " << n.n.forwardPass ({0, 0}) [0] << endl;
    cout << "0 xor 1 = " << n.n.forwardPass ({0, 1}) [0] << endl;
    cout << "1 xor 0 = " << n.n.forwardPass ({1, 0}) [0] << endl;
    cout << "1 xor 1 = " << n.n.forwardPass ({1, 1}) [0] << endl;
}


void loop () {

}
