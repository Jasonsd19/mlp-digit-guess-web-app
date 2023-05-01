#include <matrix.h>
#include <algorithm>
#include <random>

// Represents a hidden layer in a multilayer perceptron, consists of weights and biases.
struct hiddenLayer {
    matrix weights;
    matrix biases;
    hiddenLayer(matrix weights, matrix biases): weights(weights), biases(biases) {}
};

// This represents a multilayer perceptron. It must have one input layer, one output layer,
// and an arbitrary number of hidden layers.
class MLP {
private:
    matrix inputWeights;
    matrix outputBiases;
    std::vector<hiddenLayer> hiddenLayers;

    // Dot product between weights and inputs. Biases added after.
    matrix summation(matrix weights, matrix inputs, matrix biases) {
        return matrix::matrixMultiply(weights, inputs) + biases;
    }

    // Applies the sigmoid function to every entry of the given matrix.
    // The input matrix is a weighted summation of the inputs plus biases of some arbitrary layer L.
    matrix sigmoid(matrix weightedSummation) {
        return matrix::map(weightedSummation, [](double x) { return 1.0 / (1 + std::exp(-x)); });
    }

    // Calculates the average squared error between the given predictions and labels.
    double cost(matrix predictions, matrix labels) {
        return matrix::mapSum((predictions - labels), [](double x) {return x * x;}) / predictions.getRows();
    }

public:
    // Initializes the size of each layer of the network, there must be one input layer, one output layer, and arbitrary hidden layers.
    // Also initializes the relevant weights and biases with random values derived from a uniform real distribution ranging from -1 to 1.
    MLP(int inputSize, int outputSize, std::vector<int> hiddenSizes) {
        if (hiddenSizes.size() < 1) throw std::invalid_argument("There must be at least one hidden layer.");

        double lower_bound = -1;
        double upper_bound = 1;
        std::default_random_engine re(time(0));
        std::uniform_real_distribution<double> unif(lower_bound, upper_bound);

        doubleArray_t inputWeightData, outputBiasData;

        for (int i = 0; i < hiddenSizes[0]; i++) {
            for (int j = 0; j < inputSize; j++) {
                inputWeightData.push_back(unif(re));
            }
        }
        this->inputWeights = matrix(inputWeightData, hiddenSizes[0], inputSize);

        for (int i = 0; i < hiddenSizes.size() - 1; i++) {

            doubleArray_t hiddenWeightData;
            doubleArray_t hiddenBiasData;
            for (int j = 0; j < hiddenSizes[i + 1]; j++) {
                for (int k = 0; k < hiddenSizes[i]; k++) {
                    hiddenWeightData.push_back(unif(re));
                    if (j == 0) hiddenBiasData.push_back(unif(re));
                }
            }
            hiddenLayers.push_back(hiddenLayer(matrix(hiddenWeightData, hiddenSizes[i + 1], hiddenSizes[i]), matrix(hiddenBiasData, hiddenSizes[i], 1)));
        }

        doubleArray_t hiddenWeightData;
        doubleArray_t hiddenBiasData;
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSizes.back(); j++) {
                hiddenWeightData.push_back(unif(re));
                if (j == 0) hiddenBiasData.push_back(unif(re));

            }
        }
        hiddenLayers.push_back(hiddenLayer(matrix(hiddenWeightData, outputSize, hiddenSizes.back()), matrix(hiddenBiasData, hiddenSizes.back(), 1)));


        for (int i = 0; i < outputSize; i++) {
            outputBiasData.push_back(unif(re));
        }
        this->outputBiases = matrix(outputBiasData, outputSize, 1);
    }

    // Returns the input weights as a matrix, and hidden weights as a vector of matrices.
    std::tuple<matrix, std::vector<matrix>> getWeights() {
        std::vector<matrix> hiddenWeights;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            hiddenWeights.push_back(hiddenLayers[i].weights);
        }

        return std::tuple<matrix, std::vector<matrix>>(inputWeights, hiddenWeights);
    }

    // Returns the input biases as a matrix, and hidden biases as a vector of matrices.
    std::tuple<matrix, std::vector<matrix>> getBiases() {
        std::vector<matrix> hiddenBiases;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            hiddenBiases.push_back(hiddenLayers[i].biases);
        }

        return std::tuple<matrix, std::vector<matrix>>(inputWeights, hiddenBiases);
    }

    // Sets the input weights and hidden weights to the given matrix and vector of matrices.
    void setWeights(matrix inputWeights, std::vector<matrix> hiddenWeights) {
        if (hiddenWeights.size() != hiddenLayers.size()) throw std::invalid_argument("The number of hidden layers must match the number of hidden weights.");

        this->inputWeights = inputWeights;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            hiddenLayers[i].weights = hiddenWeights[i];
        }
    }

    // Sets the output biases and hidden biases to the given matrix and vector of matrices.
    void setBiases(matrix outputBiases, std::vector<matrix> hiddenBiases) {
        if (hiddenBiases.size() != hiddenLayers.size()) throw std::invalid_argument("The number of hidden layers must match the number of hidden biases.");

        this->outputBiases = outputBiases;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            hiddenLayers[i].biases = hiddenBiases[i];
        }
    }

    // Returns a tuple containing a vector of activation matrices, and a single matrix containing the output values.
    // Takes in a m x n matrix of training inputs where m is the amount of training examples, and n is the amount of features/inputs per example.
    std::tuple<std::vector<matrix>, matrix> prediction(matrix I) {
        std::vector<matrix> hiddenActivations;

        matrix firstWS = summation(inputWeights, I, hiddenLayers[0].biases);
        matrix firstActivation = sigmoid(firstWS);
        hiddenActivations.push_back(firstActivation);

        for (int i = 0; i < hiddenLayers.size() - 1; i++) {
            matrix hiddenWS = summation(hiddenLayers[i].weights, hiddenActivations[i], hiddenLayers[i + 1].biases);
            matrix hiddenActivation = sigmoid(hiddenWS);
            hiddenActivations.push_back(hiddenActivation);
        }

        matrix lastWs = summation(hiddenLayers.back().weights, hiddenActivations.back(), outputBiases);
        matrix lastActivation = sigmoid(lastWs);

        return std::tuple<std::vector<matrix>, matrix>(hiddenActivations, lastActivation);
    }

    // Takes in a m x n matrix of training inputs where m is the amount of training examples, and n is the amount of features per example. Takes in matrix of training labels with m rows.
    // This function 'trains' the model by making predictions via forward propagation, and then backpropagating the error and adjusting the weights and biases to minimize the error between its prediction and the real value.
    // Returns the updated weights and biases, and an array of error values corresponding to each epoch.
    std::tuple<matrix, matrix, std::vector<hiddenLayer>, doubleArray_t> train(matrix I, matrix L, double learningRate = 0.01, double maxEpochs = 100, double errorCutoff = 1e-3) {
        int epoch = 0;
        doubleArray_t errors;
        double error = 1000.0;

        // Define threshold to stop the training process
        while (epoch <= maxEpochs && error > errorCutoff) {
            double loss = 0.0;

            for (int i = 0; i < I.getRows(); i++) {
                matrix testData = matrix::transpose(matrix::getRow(I, i));
                matrix testLabel = matrix::transpose(matrix::getRow(L, i));

                // ---------- Forward propagation ----------
                auto [hiddenAs, lastA] = prediction(testData);

                // ---------- Calculate error/loss ----------
                loss += cost(lastA, testLabel);


                // ---------- Back propagation to update weights and biases ----------

                matrix lastPartialDerivative = matrix::scalarMultiply(lastA - testLabel, 2.0) * (lastA * (matrix::map(lastA, [](double x) { return 1.0 - x; })));

                matrix lastWeightGradient = matrix::matrixMultiply(lastPartialDerivative, matrix::transpose(hiddenAs.back()));

                std::vector<matrix> hiddenPartialDerivatives;
                std::vector<matrix> hiddenWeightGradients;
                hiddenPartialDerivatives.push_back(lastPartialDerivative);
                hiddenWeightGradients.push_back(lastWeightGradient);

                for (int i = hiddenAs.size() - 1; i > 0; i--) {
                    matrix hiddenPartialDerivative = matrix::matrixMultiply(matrix::transpose(hiddenLayers[i].weights), hiddenPartialDerivatives[hiddenAs.size() - 1 - i]) * (hiddenAs[i] * (matrix::map(hiddenAs[i], [](double x) { return 1.0 - x; })));
                    matrix hiddenWeightGradient = matrix::matrixMultiply(hiddenPartialDerivative, matrix::transpose(hiddenAs[i - 1]));

                    hiddenPartialDerivatives.push_back(hiddenPartialDerivative);
                    hiddenWeightGradients.push_back(hiddenWeightGradient);
                }

                matrix firstPartialDerivative = matrix::matrixMultiply(matrix::transpose(hiddenLayers[0].weights), hiddenPartialDerivatives.back()) * (hiddenAs[0] * (matrix::map(hiddenAs[0], [](double x) { return 1.0 - x; })));
                hiddenPartialDerivatives.push_back(firstPartialDerivative);
                matrix firstWeightGradient = matrix::matrixMultiply(firstPartialDerivative, matrix::transpose(testData));

                outputBiases = outputBiases - matrix::scalarMultiply(lastPartialDerivative, learningRate);
                for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
                    hiddenLayers[i].weights = hiddenLayers[i].weights - matrix::scalarMultiply(hiddenWeightGradients[hiddenLayers.size() - 1 - i], learningRate);
                    hiddenLayers[i].biases = hiddenLayers[i].biases - matrix::scalarMultiply(hiddenPartialDerivatives[hiddenLayers.size() - i], learningRate);
                }
                inputWeights = inputWeights - matrix::scalarMultiply(firstWeightGradient, learningRate);
            }

            error = loss / I.getRows();
            errors.push_back(error);
            std::cout << "Epoch: " << epoch << ". Loss: " << error << "." << std::endl;
            epoch++;
        }

        return std::tuple<matrix, matrix, std::vector<hiddenLayer>, doubleArray_t>(inputWeights, outputBiases, hiddenLayers, errors);
    }

    // Tests the trained model against the provided test data and labels, and prints out the accuracy.
    void test(matrix I, matrix L) {
        int correct = 0;
        for (int i = 0; i < I.getRows(); i++) {
            matrix testData = matrix::transpose(matrix::getRow(I, i));
            matrix testLabel = matrix::transpose(matrix::getRow(L, i));

            auto [hiddenAs, lastA] = prediction(testData);

            // This section of the code determines the accuracy, it's arbitrary and
            // is upto the user to decide what constitutes a correct prediction.
            int confidenceVal = -1;
            int labelVal = 0;
            double confidence = 0.0;
            for (int i = 0; i < lastA.getRows(); i++) {
                if ((int)testLabel(i, 0) == 1) labelVal = i;
                if (lastA(i, 0) > confidence) {
                    confidence = lastA(i, 0);
                    confidenceVal = i;
                }
            }

            if (confidenceVal == labelVal) correct++;
        }

        std::cout << "Accuracy: " << (double)correct / I.getRows() * 100 << "%" << std::endl;
    }

};