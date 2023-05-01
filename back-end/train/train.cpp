#include <multilayerPerceptron.cpp>
#include <csvParser.cpp>
#include <iostream>
#include <fstream>

// Helper function for writing weights and biases to text file
void writeToFile(std::string fileName, matrix inputWeights, matrix outputBiases, std::vector<hiddenLayer> hiddenLayers) {
    std::ofstream outFile(fileName);

    outFile << "inputWeights" << std::endl;
    for (int i = 0; i < inputWeights.getRows(); i++) {
        for (int j = 0; j < inputWeights.getColumns(); j++) {
            outFile << inputWeights(i, j) << " ";
        }
        outFile << "\n";
    }


    for (int i = 0; i < hiddenLayers.size(); i++) {
        outFile << "\n" << "hiddenLayerWeights" << std::endl;
        for (int j = 0; j < hiddenLayers[i].weights.getRows(); j++) {
            for (int k = 0; k < hiddenLayers[i].weights.getColumns(); k++) {
                outFile << hiddenLayers[i].weights(j, k) << " ";
            }
            outFile << "\n";
        }


        outFile << "\n" << "hiddenLayerBiases" << std::endl;
        for (int j = 0; j < hiddenLayers[i].biases.getRows(); j++) {
            for (int k = 0; k < hiddenLayers[i].biases.getColumns(); k++) {
                outFile << hiddenLayers[i].biases(j, k) << " ";
            }
            outFile << "\n";
        }
    }

    outFile << "\n" << "outputBiases" << std::endl;
    for (int i = 0; i < outputBiases.getRows(); i++) {
        for (int j = 0; j < outputBiases.getColumns(); j++) {
            outFile << outputBiases(i, j) << " ";
        }
        outFile << "\n";
    }

}

int main() {
    // Read in training and test data/labels
    auto [trainInputs, trainLabels] = csv::read_data("../../train/mnist_train.csv", 10);
    auto [testInputs, testLabels] = csv::read_data("../../train/mnist_test.csv", 10);

    // Normalize data
    trainInputs = matrix::scalarMultiply(trainInputs, 1.0 / 255.0);
    testInputs = matrix::scalarMultiply(testInputs, 1.0 / 255.0);

    // Initialize model
    MLP model = MLP(784, 10, std::vector<int>{392, 196, 98, 49, 24});

    // Train model
    auto [inputWeights, outputBiases, hiddenLayers, _] = model.train(trainInputs, trainLabels, 0.05, 100, 0.0005);

    // Test model
    model.test(testInputs, testLabels);

    // Write weights and biases to text file
    writeToFile("../../weights/784-392-196-98-49-24-10.txt", inputWeights, outputBiases, hiddenLayers);
}