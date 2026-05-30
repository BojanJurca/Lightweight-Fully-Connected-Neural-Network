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
#endif


// ----- quantization -----

// 1️⃣ Provide quantization data type for weights and biases, for example:
class Q {
    private:
        int8_t value;

        // 2️⃣ Define quantization data type range
        static constexpr float MAX_ABS_Q_VAL = 4.0f; // all inputs to quantization are expected to be in [-MAX_ABS_Q_VAL .. MAX_ABS_Q_VAL] 

    public:
        // basic constructor
        __attribute__((always_inline)) inline Q () : value (0) {}

        // constructor from any arithmetic type
        template<typename T> __attribute__((always_inline)) inline Q (T x) {
            int16_t t = (int16_t) (x * 32); // 32 = (128.0f / MAX_ABS_Q_VAL)
            if (t > 127) {
                t = 127;
                cout << "quantization OVERFLOW!\n";
            } else if (t < -128) {
                t = -128;
                cout << "quantization UNDERFLOW\n";
            }
            value = (int8_t) t;
        }

        // conversion back to float
        __attribute__((always_inline)) inline operator float () const {
            // instead of simply returning 
            // return (float) value * (4.0f / 128.0f);
            // which is comprehensive but slow let's do
            // it fast and dirty way

            int8_t b = value;

            uint16_t sign = b >> 7;

            if (sign) {
                b = (b ^ (b >> 7)) - (b >> 7); // fast bitwise b *= -1
                sign = 0b1000000000000000; // align sign bit to the first place of the first 16 bits of float
            }

            uint16_t exponent;
            uint16_t mantissa;

            // the placement of the first bit set determines the exponent, all the following bits are the beginning of mantissa
            if (b & 0b01000000) {           // 64 .. 127
                exponent = 128 << 7;                            // exponent = 128, align it to correct place in the first 16 bits of float
                mantissa = ((uint16_t) (b & 0b00111111)) << 1;  // mantissa is all that follows the first bit, align it to correct place in the first 16 bits of float
            } else if (b & 0b00100000) {    // 32 .. 63
                exponent = 127 << 7;                            // exponent = 127, align it to correct place in the first 16 bits of float
                mantissa = ((uint16_t) (b & 0b00011111)) << 2;  // mantissa is all that follows the first bit, align it to correct place in the first 16 bits of float
            } else if (b & 0b00010000) {    // 16 .. 31
                exponent = 126 << 7;                            // exponent = 126, align it to correct place in the first 16 bits of float
                mantissa = ((uint16_t) (b & 0b00001111)) << 3;  // mantissa is all that follows the first bit, align it to correct place in the first 16 bits of float
            } else if (b & 0b00001000) {    // 8 .. 15
                exponent = 125 << 7;                            // exponent = 125, align it to correct place in the first 16 bits of float
                mantissa = ((uint16_t) (b & 0b00000111)) << 4;  // mantissa is all that follows the first bit, align it to correct place in the first 16 bits of float
            } else if (b & 0b00000100) {    // 4 .. 7
                exponent = 124 << 7;                            // exponent = 124, align it to correct place in the first 16 bits of float
                mantissa = ((uint16_t) (b & 0b00000011)) << 5;  // mantissa is all that follows the first bit, align it to correct place in the first 16 bits of float
            } else if (b & 0b00000010) {    // 2, 3
                exponent = 123 << 7;                            // exponent = 123, align it to correct place in the first 16 bits of float
                mantissa = ((uint16_t) (b & 0b00000001)) << 6;  // mantissa is all that follows the first bit, align it to correct place in the first 16 bits of float
            } else if (b & 0b00000001) {    // 1
                exponent = 122 << 7;                            // = exponent = 122, align it to correct place in the first 16 bits of float
                mantissa = 0;
            } else {                        // 0
                exponent = 0;
                mantissa = 0;
            }

            // construct 32 bits of float
            struct __attribute__((packed)) fBits_t {
                uint16_t bits2; // LSB (little endian puts LSB in front of MSB) - last 16 bits of mantissa are always 0
                uint16_t bits1; // MSB - sign (1 bit), exponet (8 bits), first 7 bits of mantissa
            };
            union {
                fBits_t fBits;
                float f;
            } u;

            u.fBits = {0, (uint16_t) (sign | exponent | mantissa)};
            return u.f;
        }

        // = operator for any arithmetic type
        template<typename T>__attribute__((always_inline)) inline Q& operator = (const T& x) {
            int16_t t = (int16_t) (x * 32); // 32 = (128.0f / MAX_ABS_Q_VAL)
            if (t > 127) {
                t = 127;
                cout << "quantization OVERFLOW!\n";
            } else if (t < -128) {
                t = -128;
                cout << "quantization UNDERFLOW\n";
            }
            value = (int8_t) t;
            return *this;
        }

        // -= operator for any arithmetic type
        template<typename T> __attribute__((always_inline)) inline Q& operator -= (T other) {
            Q rhs {other};
            value -= rhs.value;
            return *this;
        }
};


// 3️⃣ Tell quantizedNeuralNetwork_t to use this data type for weights and biases
#define nnQ_t Q

// 4️⃣ Suppose we have already trained neural network
neuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> neuralNetwork;

// 5️⃣ Quantize neuralNetwork
#include <QuantizedNeuralNetwork.hpp>
quantizedNeuralNetworkLayer_t<2, FastTanh, 2, /* add more layers if needed */ FastTanh, 1> quantizedNeuralNetwork;


void setup () {
    cinit ();
    cout << setprecision (4) << fixed;

    // load pre-trained neural network model
    neuralNetwork = {0x1.099fp+0f,0x1.072ed4p+0f,-0x1.14b4fcp+1f,-0x1.0ef47ep+1f,-0x1.71b4bp+0f,0x1.60bf78p-1f,-0x1.1519e4p+1f,-0x1.fdc5b8p+0f,-0x1.8f5b94p-1f};
    // quantize the model
    quantizedNeuralNetwork = neuralNetwork;
    // output quantized model so we can use it later skipping the quantization from neuralNetwork ...
    cout << "quantizedNeuralNetwork =\n" << quantizedNeuralNetwork;
    // ... like this:
    #ifdef ARDUINO_ARCH_AVR
        quantizedNeuralNetwork = {8225,-16965,5842,-15941,232}; // {11308,-22876,7619,-21340,-8448}; // int16_t initializer list for AVR boards, 
    #else
        quantizedNeuralNetwork = {-1111810015,-1044703534,232}; // int32_t initializer list for others
    #endif

cout << "quantizedNeuralNetwork =\n" << quantizedNeuralNetwork;    
cout << "AAAAAAAAAAAAAAAAAAAAAAAAAA\n";


    // 6️⃣ Compare neuralNetwork and quantizedNeuralNetwork

    cout << "\n----- neuralNetwork -----\n\n";
    cout << "0 xor 0 = " << neuralNetwork.forwardPass ({0, 0}) [0] << endl;
    cout << "0 xor 1 = " << neuralNetwork.forwardPass ({0, 1}) [0] << endl;
    cout << "1 xor 0 = " << neuralNetwork.forwardPass ({1, 0}) [0] << endl;
    cout << "1 xor 1 = " << neuralNetwork.forwardPass ({1, 1}) [0] << endl;

    cout << "\nsizeof (neuralNetwork) = "  << sizeof (neuralNetwork) << endl;
    unsigned long startMillis = millis ();
    for (int i = 0; i < 1000; i++) {
        neuralNetwork.forwardPass ({0, 0});
        neuralNetwork.forwardPass ({0, 1});
        neuralNetwork.forwardPass ({1, 0});
        neuralNetwork.forwardPass ({1, 1});
    }
    unsigned long endMillis = millis ();
    cout << "\n1000 x 4 forwardPass-es: " << endMillis - startMillis << " ms\n";

    cout << "\n----- quantizedNeuralNetwork -----\n\n";
    cout << "0 xor 0 = " << quantizedNeuralNetwork.forwardPass ({0, 0}) [0] << endl;
    cout << "0 xor 1 = " << quantizedNeuralNetwork.forwardPass ({0, 1}) [0] << endl;
    cout << "1 xor 0 = " << quantizedNeuralNetwork.forwardPass ({1, 0}) [0] << endl;
    cout << "1 xor 1 = " << quantizedNeuralNetwork.forwardPass ({1, 1}) [0] << endl;

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
}

void loop () {

}
