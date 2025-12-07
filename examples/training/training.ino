/*
 *      You may consider using a computer to train the neural network
 */


// platform abstraction 
#ifdef ARDUINO                  // Arduino build requires LightwaightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
    #include <array.hpp>
    #include <iostream.hpp>
#else                           // standard C++ build
    #include <array>
    #include <iostream>
    #include <iomanip>
    using namespace std;
    void setup ();
    void loop ();
    int main () { setup (); while (true) loop (); return 0; }        
    #define cinit(...)          // compiles to nothing
#endif


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
    cout << fixed << setprecision (6);
    
    // store the best training result
    float lowestLoss = 1.0f / 0.0f;
    decltype (neuralNetwork) bestModel;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {

        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (int e = 0; e < 10000; e++) {
            float loss = 0.0f;

            // backwardPropagation returns the output-layer error, computed as the
            // difference between the expected and the predicted values.
            // The error is calculated as: error = 1/2 sqrt ∑ (expected − predicted)².
            // This value is provided only as informational feedback on the progress of training.

            loss += neuralNetwork.backwardPropagation ({0, 0}, {0});
            loss += neuralNetwork.backwardPropagation ({0, 1}, {1});
            loss += neuralNetwork.backwardPropagation ({1, 0}, {1});
            loss += neuralNetwork.backwardPropagation ({1, 1}, {0});

            // keep track of the best model so far           
            if (loss < lowestLoss) {
                lowestLoss = loss;
                bestModel = neuralNetwork;
            } 
            
            if (e % 1000 == 0)
                cout << "   t = " << t << "   e = " << e << ", loss = " << loss << ", best = " << lowestLoss << endl; 
        }
        cout << "   t = " << t << ", best = " << lowestLoss << endl; 

        // repeat training from different seeds
        srand (static_cast<unsigned> (time (nullptr)));
        neuralNetwork.randomize ();
    }


    // restore the best model
    neuralNetwork = bestModel;
    cout << "the best model = " << hexfloat << neuralNetwork << fixed << endl << endl; // for example: {0x1.0cf11ap+1f,0x1.fac59cp+0f,0x1.01d206p+0f,0x1.f5a018p-1f,-0x1.3e38c2p-1f,-0x1.5fe124p+0f,0x1.121f7cp+1f,-0x1.25b456p+1f,-0x1.bf2034p-1f}


    cout << "the neural network is trained and ready\n\n";

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
