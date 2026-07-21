// ----- platform abstraction  -----

#ifdef ARDUINO                  // Arduino build requires LightwaightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
    #include <array.hpp>
    #include <iostream.hpp>
    #include <LightweightNeuralNetwork.hpp>
    #define srand(X) randomSeed(X)
    #ifdef ARDUINO_ARCH_AVR     // Arduino AVR
        unsigned long time (void *p) { p = p; return millis (); } // introduce time function only for the purpose of srand (time (NULL)) would work on AVR boards as well
    #endif
#else                           // standard C++ build
    #include <array>
    #include <iostream>
    #include <iomanip>
    #include <LightweightNeuralNetwork.hpp>
    using namespace std;
    void setup ();
    void loop ();
    int main () { setup (); while (true) loop (); return 0; }        
    #define cinit(...)          // compiles to nothing
    #include <chrono>
    unsigned long millis() {
        using namespace std::chrono;
        return duration_cast<milliseconds> (steady_clock::now ().time_since_epoch ()).count ();
    }
#endif


void testSigmoid () {

    cout << "\n----- Sigmoid training -----\n";
    
    //                   .--- the number inputs
    //                   |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     .--- the number of neurons in the first layer
    //                   |      |     |                                    .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     |                                    |     .--- the number of neurons in the output layer = the number of outputs
    //                   |      |     |                                    |     |
    neuralNetworkLayer_t<2, Sigmoid, 2, /* add more layers if needed */ Sigmoid, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 100000; e++) {
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
*/

    cout << "the best model = {\n" << neuralNetwork << "};\n";
    
    bestModel = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: Sigmoid -----
        //    --- weights ---
            -3.9454f, -3.9500f, 
            -5.7542f, -5.7809f, 
        //    --- biases ---
            5.8198f, 2.1784f, 
        //    --- min: -5.7809, max: 5.8198 --- in case of quantization use at least Q<8>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: Sigmoid -----
        //    --- weights ---
            7.6796f, -7.9929f, 
        //    --- biases ---
            -3.5184f
        //    --- min: -7.9929, max: 7.6796 --- in case of quantization use at least Q<8>
    };
    
    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- Sigmoid model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- Sigmoid inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testFastSigmoid () {

    cout << "\n----- FastSigmoid training -----\n";
    
    //                   .--- the number inputs
    //                   |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     .--- the number of neurons in the first layer
    //                   |      |     |                                            .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     |                                            |     .--- the number of neurons in the output layer = the number of outputs
    //                   |      |             |                                    |     |
    neuralNetworkLayer_t<2, FastSigmoid, 2, /* add more layers if needed */ FastSigmoid, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 100000; e++) {
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
*/

    cout << "the best model = {\n" << neuralNetwork << "};\n";

    bestModel = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: FastSigmoid -----
        //    --- weights ---
            4.2421f, -4.4985f, 
            4.7259f, -4.4948f, 
        //    --- biases ---
            -2.3593f, 2.3431f, 
        //    --- min: -4.4985, max: 4.7259 --- in case of quantization use at least Q<8>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: FastSigmoid -----
        //    --- weights ---
            8.9860f, -8.4355f, 
        //    --- biases ---
            3.2067f
        //    --- min: -8.4355, max: 8.9860 --- in case of quantization use at least Q<16>
    };
      
    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- FastSigmoid model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- FastSigmoid inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testHardSigmoid () {

    cout << "\n----- HardSigmoid training -----\n";
    
    //                   .--- the number inputs
    //                   |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     .--- the number of neurons in the first layer
    //                   |      |     |                                            .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     |                                            |     .--- the number of neurons in the output layer = the number of outputs
    //                   |      |             |                                    |     |
    neuralNetworkLayer_t<2, HardSigmoid, 2, /* add more layers if needed */ HardSigmoid, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 1000000; e++) {
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
*/

    cout << "the best model = {\n" << bestModel << "};\n";

    bestModel =  {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: HardSigmoid -----
        //    --- weights ---
            -3.6181f, -3.6181f, 
            5.0003f, 5.0004f, 
        //    --- biases ---
            4.7372f, -2.5002f, 
        //    --- min: -3.6181, max: 5.0004 --- in case of quantization use at least Q<8>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: HardSigmoid -----
        //    --- weights ---
            6.9082f, 6.9071f, 
        //    --- biases ---
            -9.4078f
        //    --- min: -9.4078, max: 6.9082 --- in case of quantization use at least Q<16>
    };
              
    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- HardSigmoid model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- HardSigmoid inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testPQSigmoid () {

    cout << "\n----- PQSigmoid training -----\n";
    
    //                   .--- the number inputs
    //                   |      .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     .--- the number of neurons in the first layer
    //                   |      |     |                                        .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |      |     |                                        |     .--- the number of neurons in the output layer = the number of outputs
    //                   |      |     |                                        |     |
    neuralNetworkLayer_t<2, PQSigmoid, 2, /* add more layers if needed */ PQSigmoid, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 100000; e++) {
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
*/

    cout << "the best model = {\n" << bestModel << "};\n";

    bestModel = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: PQSigmoid -----
        //    --- weights ---
            -3.0894f, 3.1091f, 
            -2.6812f, 2.6451f, 
        //    --- biases ---
            1.5478f, -1.4845f, 
        //    --- min: -3.0894, max: 3.1091 --- in case of quantization use at least Q<4>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: PQSigmoid -----
        //    --- weights ---
            -3.9493f, 4.2528f, 
        //    --- biases ---
            1.8967f
        //    --- min: -3.9493, max: 4.2528 --- in case of quantization use at least Q<8>
    };
      
    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- PQSigmoid model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- PQSigmoid inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testReLU () {

    cout << "\n----- ReLU training -----\n";
    
    //                   .--- the number inputs
    //                   |  .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |     .--- the number of neurons in the first layer
    //                   |  |     |                                  .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |     |                                  |     .--- the number of neurons in the output layer = the number of outputs
    //                   |  |     |                                  |     |
    neuralNetworkLayer_t<2, ReLU, 2, /* add more layers if needed */ ReLU, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 10000; e++) {
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
*/

    cout << "the best model = {\n" << bestModel << "};\n";

    bestModel =  {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: ReLU -----
        //    --- weights ---
            -0.8198f, 0.8198f, 
            1.2909f, -1.1472f, 
        //    --- biases ---
            0.0000f, -0.1438f, 
        //    --- min: -1.1472, max: 1.2909 --- in case of quantization use at least Q<2>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: ReLU -----
        //    --- weights ---
            1.2111f, 0.8664f, 
        //    --- biases ---
            0.0040f
        //    --- min: 0.0040, max: 1.2111 --- in case of quantization use at least Q<2>
    };

    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- ReLU model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- ReLU inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testTanh () {

    cout << "\n----- Tanh training -----\n";
    
    //                   .--- the number inputs
    //                   |  .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |     .--- the number of neurons in the first layer
    //                   |  |     |                                  .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |     |                                  |     .--- the number of neurons in the output layer = the number of outputs
    //                   |  |     |                                  |     |
    neuralNetworkLayer_t<2, Tanh, 2, /* add more layers if needed */ Tanh, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 100000; e++) {
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
*/

    cout << "the best model = {\n" << bestModel << "};\n";

    bestModel = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: Tanh -----
        //    --- weights ---
            1.2926f, 1.1924f, 
            -2.9151f, -2.2225f, 
        //    --- biases ---
            -1.8340f, 0.8524f, 
        //    --- min: -2.9151, max: 1.2926 --- in case of quantization use at least Q<4>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: Tanh -----
        //    --- weights ---
            -2.6660f, -2.3991f, 
        //    --- biases ---
            -0.8722f
        //    --- min: -2.6660, max: -0.8722 --- in case of quantization use at least Q<4>
    };

    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- Tanh model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- Tanh inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testFastTanh () {

    cout << "\n----- FastTanh training -----\n";
    
    //                   .--- the number inputs
    //                   |  .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |         .--- the number of neurons in the first layer
    //                   |  |         |                                      .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |         |                                      |     .--- the number of neurons in the output layer = the number of outputs
    //                   |  |         |                                      |     |
    neuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 100000; e++) {
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
*/

    cout << "the best model = {\n" << bestModel << "};\n";

    bestModel = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: FastTanh -----
        //    --- weights ---
            -2.2071f, -2.5444f, 
            1.0820f, 1.1404f, 
        //    --- biases ---
            0.8130f, -1.5762f, 
        //    --- min: -2.5444, max: 1.1404 --- in case of quantization use at least Q<4>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: FastTanh -----
        //    --- weights ---
            -2.2208f, -2.4621f, 
        //    --- biases ---
            -0.8024f
        //    --- min: -2.4621, max: -0.8024 --- in case of quantization use at least Q<4>
    };


    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- FastTanh model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- FastTanh inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void testPQTanh () {

    cout << "\n----- PQTanh training -----\n";
    
    //                   .--- the number inputs
    //                   |  .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |       .--- the number of neurons in the first layer
    //                   |  |       |                                  .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
    //                   |  |       |                                  |       .--- the number of neurons in the output layer = the number of outputs
    //                   |  |       |                                  |       |
    neuralNetworkLayer_t<2, PQTanh, 2, /* add more layers if needed */ PQTanh, 1> neuralNetwork;

    // store the best training result
    decltype (neuralNetwork) bestModel;

/* training can last a while on Arduino, skip it for now and just load already trained model later ...

    float lowestLoss = 1.0f / 0.0f;

    // perform, say, 20 independent training runs starting from different random initializations, most of which will likely converge to different local minima
    for (int t = 0; t < 20; t++) {
        // perform, say, 10000 (= epoch) gradient descent iterations to reach a local minimum
        for (long e = 0; e < 100000; e++) {
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
*/

    cout << "the best model = {\n" << bestModel << "};\n";

    bestModel = {
        // ----- layer inputs: 2, outputs (neurons): 2, activation: PQTanh -----
        //    --- weights ---
            2.4828f, 2.2623f, 
            -1.0830f, -1.0364f, 
        //    --- biases ---
            -0.7516f, 1.3891f, 
        //    --- min: -1.0830, max: 2.4828 --- in case of quantization use at least Q<4>
        // ----- layer inputs: 2, outputs (neurons): 1, activation: PQTanh -----
        //    --- weights ---
            2.1039f, 2.2532f, 
        //    --- biases ---
            -0.7589f
        //    --- min: -0.7589, max: 2.2532 --- in case of quantization use at least Q<4>
    };


    // restore the best model
    neuralNetwork = bestModel;


    cout << "\n----- PQTanh model test -----\n";

    float output;
    float error;
    float loss = 0;

    output = neuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = neuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;


    cout << "\n----- PQTanh inference -----\n";

    // measure speed
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "1000 x 4 inferences: " << endMillis - startMillis << " ms\n";
}


void setup () {
    cinit ();
    cout << fixed << setprecision (4);

    testSigmoid ();

    testFastSigmoid ();
    
    testHardSigmoid ();
    
    testPQSigmoid ();
    
    testReLU ();
    
    testTanh ();
    
    testFastTanh ();

    testPQTanh ();
}

void loop () {
    #ifndef ARDUINO
        exit (0);
    #endif
}
