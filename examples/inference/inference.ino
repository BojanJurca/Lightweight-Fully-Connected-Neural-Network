#include <array.hpp>        // LightweightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
#include <iostream.hpp>     // LightweightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
#include <LightweightNeuralNetwork.hpp>


// ----- the neural network -----
    
    //                   .--- the number inputs
    //                   |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     .--- the number of neurons in the first layer
    //                   |      |     |                                      .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     |                                      |        .--- the number of neurons in the output layer = the number of outputs
    //                   |      |     |                                      |        |
    neuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastSigmoid, 1> neuralNetwork;
    // at this point neuralNetwork is initialized with random weights and biases and it is ready for training
    // - you can either start training it and export the trained model when fiished
    // - or you can load already trained model that is cappable of making usable outputs 


void setup () {

    cinit ();

    // load pre-trained model
    neuralNetwork = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: FastTanh -----
        //    --- weights ---
            -4.4286f, -4.4286f, 
            3.8355f, 3.8355f, 
        //    --- biases ---
            2.1577f, -5.6935f, 
        //    --- min: -5.6935, max: 3.8355 --- in case of quantization use at least Q<8>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: FastSigmoid -----
        //    --- weights ---
            -10.2262f, -10.2330f, 
        //    --- biases ---
            -9.0623f
        //    --- min: -10.2330, max: -9.0623 --- in case of quantization use at least Q<16>
    };

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
