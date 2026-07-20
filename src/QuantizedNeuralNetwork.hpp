/*

    QuantizedNeuralNetwork.hpp

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Lightweight-Fully-Connected-Neural-Network

    QuantizedNeuralNetwork provides post training quantization for LightweightNeuralNetwork 
    with user defined Quant quantization type for weights and biases.

    Bojan Jurca, Aug 12, 2026

*/


#ifndef __Q_NEURAL_NETWORK_HPP__
    #define __Q_NEURAL_NETWORK_HPP__

    #include "LightweightNeuralNetwork.hpp"
    

    // ----- quantization -----

    // Uniform 8‑bit quantization with dynamic range n; n = 1, 2, 4, 8, ...
    // Q<1> → uniform 8‑bit quantization, range = [-1, +1)
    template<int N>
    class Q {
        private:
            
            static constexpr float __scale__ = 128.0f / N; // scale factor
        
        public:
        
            int8_t raw;
    
            __attribute__((always_inline)) inline Q () : raw (0) {}
    
            // conversion to int8_t raw 
            template<typename T>
            __attribute__((always_inline)) inline Q& operator = (const T& x) {
                int16_t t = static_cast<int16_t>(x * __scale__);
        
                if (t > 127) {
                    cout << "quantization OVERFLOW while trying to quantize " << x << ", raw = " << t << " to Q<" << N << ">\n";
                    t = 127;
                } else if (t < -128) {
                    cout << "quantization UNDERFLOW while trying to quantize " << x << ", raw = " << t << " to Q<" << N << ">\n";
                    t = -128;
                }
        
                raw = static_cast<int8_t>(t);
                return *this;
            }
    
            template<typename T>
            __attribute__((always_inline)) inline Quant& operator -= (T other) {
                // Quant rhs {(Quant) other};
                Quant q;
                q = other;
                raw -= q.raw;
                return *this;
            }

            template<typename T>
            __attribute__((always_inline)) inline Quant& operator += (T other) {
                //Quant rhs {(Quant) other};
                Quant q;
                q = other;
                raw += q.raw;
                return *this;
            }
    
            // conversion of int8_t raw to float
    
        private:
    
            static constexpr int log2 (unsigned x) {
                return (x > 1) ? 1 + log2 (x >> 1) : 0;
            }
        
        public:
        
            __attribute__((always_inline)) inline operator float () const {
                
                // instead of simply returning (float) raw / __scale__ which is slow
                // we'll do it a faster way:
    
                int8_t value = raw;
                uint16_t sign = value >> 7;
                if (sign) {
                    if (value == -128) return -N; // -128 -> -N
                    value = (value ^ (value >> 7)) - (value >> 7); // else fast raw *= -1 and treat sign as a separate value
                    sign = 0b1000000000000000;
                }
                uint16_t exponent;
                uint16_t mantissa;
    
                     if ((uint8_t) value & 0b01000000) { exponent = (0b01111110 + log2 (N)) << 7; mantissa = (uint16_t) ((uint8_t) value & 0b00111111) << 1; } // value ∈ [64, 127]
                else if ((uint8_t) value & 0b00100000) { exponent = (0b01111101 + log2 (N)) << 7; mantissa = (uint16_t) ((uint8_t) value & 0b00011111) << 2; } // value ∈ [32, 63]
                else if ((uint8_t) value & 0b00010000) { exponent = (0b01111100 + log2 (N)) << 7; mantissa = (uint16_t) ((uint8_t) value & 0b00001111) << 3; } // value ∈ [16, 31]
                else if ((uint8_t) value & 0b00001000) { exponent = (0b01111011 + log2 (N)) << 7; mantissa = (uint16_t) ((uint8_t) value & 0b00000111) << 4; } // value ∈ [8, 15]
                else if ((uint8_t) value & 0b00000100) { exponent = (0b01111001 + log2 (N)) << 7; mantissa = (uint16_t) ((uint8_t) value & 0b00000011) << 5; } // value ∈ [4, 7]
                else if ((uint8_t) value & 0b00000010) { exponent = (0b01111001 + log2 (N)) << 7; mantissa = (uint16_t) ((uint8_t) value & 0b00000001) << 6; } // value ∈ [2, 3]
                else if ((uint8_t) value & 0b00000001) { exponent = (0b01111000 + log2 (N)) << 7; mantissa = 0; } // value = 1
                else return 0; // value = 0
    
                struct __attribute__((packed)) fBits_t {
                    uint16_t bits2;
                    uint16_t bits1;
                };
        
                union {
                    fBits_t fBits;
                    float f;
                } u;
    
                u.fBits = {0, (uint16_t) (sign | exponent | mantissa)}; // bits2 (lower bits of mantissa) are always 0 
                return u.f;
            }
    
    };


    // ----- quantized neural network -----

    // define quantization data type
    #ifndef Quant
        #define Quant Q<1> // use float as default data type (no quantization)
    #endif


    // basic neuralNetwork_t class template, not used but needed by C++ compiler
        template <size_t... sizes> 
        class quantizedNeuralNetworkLayer_t;


    // hidden layers
        template <size_t inputCount, size_t activationFunction, size_t neuronCount, size_t... sizes> 
        class quantizedNeuralNetworkLayer_t<inputCount, activationFunction, neuronCount, sizes...> {

                // data structures needed for this layer: weight and bias
                Quant weight [neuronCount][inputCount];
                Quant bias [neuronCount];

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
                    os << "\n#ifdef ARDUINO_ARCH_AVR // int16_t initializer list\n   ";
                    for (size_t i = 0; i < bytesToWrite; i += 2) {
                        int16_t v = p [i];
                        if (i + 1 < bytesToWrite) 
                            v |= (int16_t) p [i + 1] << 8; // little endian int16_t
                        os << v;
                        if (i + 2 < bytesToWrite) os << ",";
                    }

                    // int32_t
                    os << "\n#else // int32_t initializer list\n   ";
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
                    os << "\n#endif\n";
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
                    Quant *q = (Quant *) this;
                    for (size_t i = 0; i < sizeof (other) / sizeof (float); i++)
                        q [i] = p [i];
                    return *this;
                }

        };


    // output layer
        template <size_t inputCount, size_t activationFunction, size_t neuronCount> 
        class quantizedNeuralNetworkLayer_t<inputCount, activationFunction, neuronCount> {

                // data structures needed for this layer: weight and bias
                Quant weight [neuronCount][inputCount];
                Quant bias [neuronCount];
                
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
                    os << "\n#ifdef ARDUINO_ARCH_AVR // int16_t initializer list\n   ";
                    for (size_t i = 0; i < bytesToWrite; i += 2) {
                        int16_t v = p [i];
                        if (i + 1 < bytesToWrite) 
                            v |= (int16_t) p [i + 1] << 8; // little endian int16_t
                        os << v;
                        if (i + 2 < bytesToWrite) os << ",";
                    }

                    // int32_t
                    os << "\n#else // int32_t initializer list\n   ";
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
                    os << "\n#endif\n";
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
                    Quant *q = (Quant *) this;
                    for (size_t i = 0; i < sizeof (other) / sizeof (float); i++)
                        q [i] = p [i];
                    return *this;
                }

        };

#endif