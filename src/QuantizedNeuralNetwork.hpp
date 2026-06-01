/*

    QuantizedNeuralNetwork.hpp

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Lightweight-Fully-Connected-Neural-Network

    QuantizedNeuralNetwork provides post training quantization for LightweightNeuralNetwork 
    with user defined nnQ_t quantization type for weights and biases.

    Bojan Jurca, May 22, 2026

*/


#ifndef __Q_NEURAL_NETWORK_HPP__
    #define __Q_NEURAL_NETWORK_HPP__

    #include "LightweightNeuralNetwork.hpp"

    // define quantization data type
    #ifndef nnQ_t
        #define nnQ_t float // use float as default data type (no quantization)
    #endif


    // basic neuralNetwork_t class template, not used but needed by C++ compiler
        template <size_t... sizes> 
        class quantizedNeuralNetworkLayer_t;


    // hidden layers
        template <size_t inputCount, size_t activationFunction, size_t neuronCount, size_t... sizes> 
        class quantizedNeuralNetworkLayer_t<inputCount, activationFunction, neuronCount, sizes...> {

                // data structures needed for this layer: weight and bias
                nnQ_t weight [neuronCount][inputCount];
                nnQ_t bias [neuronCount];

                // include the next layer instance which will include the next layer itself, ...
                quantizedNeuralNetworkLayer_t<neuronCount, sizes...> nextLayer;

            public:
            
                static constexpr size_t outputCount = quantizedNeuralNetworkLayer_t<neuronCount, sizes...>::outputCount;
                using output_t = array<float, outputCount>;

                // calculates the neurons of this layer and returns the category that the input belongs to
                template<typename input_t>
                output_t forwardPass (const input_t (&input) [inputCount]) const {   
                    float neuron [neuronCount];
                    // neuron = af (w x input + bias)
                        for (size_t n = 0; n < neuronCount; n++) {
                            neuron [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                neuron [n] += weight [n][i] * input [i];
                            neuron [n] = af<activationFunction> (neuron [n]);
                        }

                    // return what the next layer thinks about the neurons clculated here
                        return nextLayer.forwardPass (neuron);
                }
            
                // make it possible to use arrays instead of C arrays
                template<typename input_t>
                __attribute__((always_inline))
                inline output_t forwardPass (const array<input_t, inputCount> input) const {   
                    return forwardPass (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()));
                }


                // export the whole model as C++ initializer list
                friend ostream& operator << (ostream& os, const quantizedNeuralNetworkLayer_t& nn) {
                    uint8_t *p = (uint8_t *) &nn;
                    size_t bytesToWrite = sizeof (nn);

                    // int16_t
                    os << "{";
                    for (size_t i = 0; i < bytesToWrite; i += 2) {
                        int16_t v = p [i];
                        if (i + 1 < bytesToWrite) 
                            v |= (int16_t) p [i + 1] << 8; // little endian int16_t
                        os << v;
                        if (i + 2 < bytesToWrite) os << ",";
                    }
                    os << "} // int16_t initializer list\n";

                    // int32_t
                    os << "{";
                    for (size_t i = 0; i < bytesToWrite; i += 4) {
                        int32_t v = p [i];
                        if (i + 1 < bytesToWrite) 
                            v |= (int32_t )p [i+1] << 8;
                        if (i + 2 < bytesToWrite) 
                            v |= (int32_t) p [i+2] << 16;
                        if (i + 3 < bytesToWrite) 
                            v |= (int32_t) p [i+3] << 24;
                        os << v;
                        if (i + 4 < bytesToWrite) os << ",";
                    }
                    os << "} // int32_t initializer list\n";
                    
                    return os;
                }


                template<size_t N>
                quantizedNeuralNetworkLayer_t& operator = (const int (&model) [N]) {
                    constexpr size_t modelBytes = sizeof(model);
                    constexpr size_t structBytes = sizeof(*this);
                    static_assert (modelBytes >= structBytes && modelBytes < structBytes + sizeof (int), "Model size does not match neural network size!");
                    memcpy (this, model, structBytes);
                    return *this;
                }

                quantizedNeuralNetworkLayer_t () = default;
                quantizedNeuralNetworkLayer_t (const quantizedNeuralNetworkLayer_t&) = default;
                quantizedNeuralNetworkLayer_t& operator = (const quantizedNeuralNetworkLayer_t&) = default;

                quantizedNeuralNetworkLayer_t& operator = (const neuralNetworkLayer_t<inputCount, activationFunction, neuronCount, sizes...>& other) {
                    float *p = (float *) &other;
                    nnQ_t *q = (nnQ_t *) this;
                    for (size_t i = 0; i < sizeof (other) / sizeof (float); i++)
                        q [i] = p [i];
                    return *this;
                }

        };


    // output layer
        template <size_t inputCount, size_t activationFunction, size_t neuronCount> 
        class quantizedNeuralNetworkLayer_t<inputCount, activationFunction, neuronCount> {

                // data structures needed for this layer: weight and bias
                nnQ_t weight [neuronCount][inputCount];
                nnQ_t bias [neuronCount];
                
            public:

                static constexpr size_t outputCount = neuronCount;
                using output_t = array<float, outputCount>;

                // calculates the output neurons of the neural network and returns the category that the input belongs to
                template<typename input_t>
                output_t forwardPass (const input_t (&input) [inputCount]) const {   
                    output_t neuron {};

                    // neuron = af (w x input + bias)
                        for (size_t n = 0; n < neuronCount; n++) {
                            neuron [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                neuron [n] += weight [n][i] * input [i];
                            neuron [n] = af<activationFunction> (neuron [n]);
                        }

                    // start returning the result through all the previous layers
                        return neuron;
                }        

                // make it possible to use arrays instead of C arrays
                template<typename input_t>
                __attribute__((always_inline))
                inline output_t forwardPass (const array<input_t, inputCount> input) const {   
                    return forwardPass (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()));
                }


                // export the whole model as C++ initializer list
                friend ostream& operator << (ostream& os, const quantizedNeuralNetworkLayer_t& nn) {
                    uint8_t *p = (uint8_t *) &nn;
                    size_t bytesToWrite = sizeof (nn);

                    // int16_t
                    os << "{";
                    for (size_t i = 0; i < bytesToWrite; i += 2) {
                        int16_t v = p [i];
                        if (i + 1 < bytesToWrite) 
                            v |= (int16_t) p [i + 1] << 8; // little endian int16_t
                        os << v;
                        if (i + 2 < bytesToWrite) os << ",";
                    }
                    os << "} // int16_t initializer list\n";

                    // int32_t
                    os << "{";
                    for (size_t i = 0; i < bytesToWrite; i += 4) {
                        int32_t v = p [i];
                        if (i + 1 < bytesToWrite) 
                            v |= (int32_t )p [i+1] << 8;
                        if (i + 2 < bytesToWrite) 
                            v |= (int32_t) p [i+2] << 16;
                        if (i + 3 < bytesToWrite) 
                            v |= (int32_t) p [i+3] << 24;
                        os << v;
                        if (i + 4 < bytesToWrite) os << ",";
                    }
                    os << "} // int32_t initializer list\n";
                    
                    return os;
                }

                template<size_t N>
                quantizedNeuralNetworkLayer_t& operator = (const int (&model) [N]) {
                    constexpr size_t modelBytes = sizeof(model);
                    constexpr size_t structBytes = sizeof(*this);
                    static_assert (modelBytes >= structBytes && modelBytes < structBytes + sizeof (int), "Model size does not match neural network size!");
                    memcpy (this, model, structBytes);
                    return *this;
                }

                quantizedNeuralNetworkLayer_t () = default;
                quantizedNeuralNetworkLayer_t (const quantizedNeuralNetworkLayer_t&) = default;
                quantizedNeuralNetworkLayer_t& operator = (const quantizedNeuralNetworkLayer_t&) = default;

                quantizedNeuralNetworkLayer_t& operator = (const neuralNetworkLayer_t<inputCount, activationFunction, neuronCount>& other) {
                    float *p = (float *) &other;
                    nnQ_t *q = (nnQ_t *) this;
                    for (size_t i = 0; i < sizeof (other) / sizeof (float); i++)
                        q [i] = p [i];
                    return *this;
                }

        };

#endif
