#include <array.hpp>
#include <iostream.hpp>
#include <LightweightNeuralNetwork.hpp>

// 1️⃣ Suppose we have already trained neural network
neuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> neuralNetwork;

// 2️⃣ Tell quantizedNeuralNetwork_t which data type to use to quantize weights and biases
// Uniform 8‑bit quantization with dynamic range N; N = 1, 2, 4, 8, ...
// Q<1> → uniform 8‑bit quantization, range = [-1, +1)
// Q<2> → uniform 8‑bit quantization, range = [-2, +2)
// ...
#define Quant Q<8> // experiment with which quantization works best for you: Q<1>, Q<2>, Q<4>, Q<8>, ...

// 3️⃣ Define quantizedNeuralNetwork with the same sructure as original neuralNetwork
#include <QuantizedNeuralNetwork.hpp>
quantizedNeuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> quantizedNeuralNetwork;


void setup () {
    cinit ();
    cout << setprecision (4) << fixed;

    // 4️⃣ load pre-trained neural network model
    neuralNetwork = {
    // ----- layer inputs: 2, outputs (neurons): 2, activation: FastTanh -----
    //    --- weights ---
            1.0376f, 1.0281f, 
            -2.1618f, -2.1168f, 
    //    --- biases ---
            -1.4442f, 0.6890f, 
    //    --- min: -2.1618, max: 1.0376 --- in case of quantization use at least Q<4>
    // ----- layer inputs: 2, outputs (neurons): 1, activation: FastTanh -----
    //    --- weights ---
            -2.1649f, -1.9913f, 
    //    --- biases ---
            -0.7800f
    //    --- min: -2.1649, max: -0.7800 --- in case of quantization use at least Q<4>
    };

    // 5️⃣ quantize the model
    quantizedNeuralNetwork = neuralNetwork;
    // output quantized model so we can use it later skipping the quantization from neuralNetwork ...
    cout << "quantizedNeuralNetwork = {" << quantizedNeuralNetwork << "};\n";
    // ... like this:
    quantizedNeuralNetwork = {
        #ifdef ARDUINO_ARCH_AVR // int16_t initializer list
            4112,-8226,3049,-7714,244
        #else // int32_t initializer list
            -539095024,-505541655,244
        #endif
    };    


    // 6️⃣ Compare output of neuralNetwork with output of quantizedNeuralNetwork

    cout << "\n----- test neuralNetwork -----\n";
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

    cout << "sizeof (neuralNetwork) = "  << sizeof (neuralNetwork) << endl;

    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "\n1000 x 4 forwardPass-es: " << endMillis - startMillis << " ms\n";


    cout << "\n----- test quantizedNeuralNetwork -----\n";
    loss = 0;

    output = quantizedNeuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "0 xor 0 = " << output << endl;
    output = quantizedNeuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "0 xor 1 = " << output << endl;
    output = quantizedNeuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    cout << "1 xor 0 = " << output << endl;
    output = quantizedNeuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "1 xor 1 = " << output << endl;
    cout << "loss = " << loss << endl;

    cout << "\nsizeof (quantizedNeuralNetwork) = "  << sizeof (quantizedNeuralNetwork) << endl;

    startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        quantizedNeuralNetwork.forwardPass ({0, 0});
        quantizedNeuralNetwork.forwardPass ({0, 1});
        quantizedNeuralNetwork.forwardPass ({1, 0});
        quantizedNeuralNetwork.forwardPass ({1, 1});
    }
    endMillis = millis ();
    cout << "\n1000 x 4 forwardPass-es: " << endMillis - startMillis << " ms\n";


    // 7️⃣ Do the post-quantization fine-tunning
    cout << "\n----- Post-Quantization Fine-Tuning -----\n\n";

    constexpr int N = sizeof (quantizedNeuralNetwork); // the number of parameters to tune
    int8_t *parameter = (int8_t *) &quantizedNeuralNetwork;
    float gradient [N];

    // calculate current loss
    loss = 0;
    output = quantizedNeuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    output = quantizedNeuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    output = quantizedNeuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
    output = quantizedNeuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
    cout << "current loss = " << loss << endl;    


    constexpr float learningStep = 0.0004; // experiment with learningStep (so that the loss decreases throu the steps to its minimum)

    for (int ts = 0; ts < 10; ts++) { // experiment with how many tunning steps work for you
        cout << "----- tunning step " << ts << " -----\n";    
        // estimate gradient
        for (int i = 0; i < N; i++) {
            gradient [i] = 0;
            float newLoss = 0;

            if (parameter [i] < 127) {
                parameter [i] += 1;
                newLoss = 0;
                output = quantizedNeuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    newLoss += error;
                output = quantizedNeuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    newLoss += error;
                output = quantizedNeuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    newLoss += error;
                output = quantizedNeuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    newLoss += error;
                gradient [i] += (newLoss - loss) / 2;
                parameter [i] -= 1;                
            }
            if (parameter [i] > -128) {
                parameter [i] -= 1;
                newLoss = 0;
                output = quantizedNeuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    newLoss += error;
                output = quantizedNeuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    newLoss += error;
                output = quantizedNeuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    newLoss += error;
                output = quantizedNeuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    newLoss += error;
                gradient [i] += (loss - newLoss) / 2;
                parameter [i] += 1;
            }
        }

        cout << "gradient = "; for (int i = 0; i < N; i ++) cout << gradient [i] << " "; cout << endl;
        
        cout << "parameters changed:";

        for (int i = 0; i < N; i ++) {
            int8_t oldValue = parameter [i];
            int newValue = (int) parameter [i] - gradient [i] * learningStep;
            if (newValue > 127) {
                cout << "post-quantization fine-tunning OVERFLOW!\n";
                newValue = 127;
            }
            if (newValue < -128) {
                cout << "post-quantization fine-tunning UNDERFLOW!\n";
                newValue = -128;
            }
            parameter [i] = (int8_t) newValue;
            
            if (oldValue != parameter [i])
                cout << " " << i;
        }
        cout << endl;

        // calculate current loss
        loss = 0;
        output = quantizedNeuralNetwork.forwardPass ({0, 0}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
        output = quantizedNeuralNetwork.forwardPass ({0, 1}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
        output = quantizedNeuralNetwork.forwardPass ({1, 0}) [0];    error = pow (output - 1, 2) / 2;    loss += error;
        output = quantizedNeuralNetwork.forwardPass ({1, 1}) [0];    error = pow (output - 0, 2) / 2;    loss += error;
        cout << "new loss = " << loss << endl;    
    }

    // output the tunned model
    cout << "\nfine-tunned quantizedNeuralNetwork /* after fine tunning */ = {" << quantizedNeuralNetwork << "};\n";
}

void loop () {

}
