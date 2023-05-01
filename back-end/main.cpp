#include <multilayerPerceptron.cpp>
#include <iostream>
#include <fstream>
#include <App.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

const unsigned short PORT = 8080;

// Helper function for reading and loading weights and biases from text file.
// Returns W1, W2, W3, B1, B2, B3 matrices if applicable.
std::tuple<matrix, matrix, std::vector<matrix>, std::vector<matrix>> readFromFile(std::string fileName) {
    std::vector<twoDimDoubleArray_t> hiddenLayerData;
    twoDimDoubleArray_t inputWeights, outputBiases, temp;

    std::ifstream inFile(fileName);

    if (inFile.is_open()) {
        std::string line;
        std::string header;

        while (getline(inFile, line)) {
            // Handle end of hidden layer data
            if (line.length() == 0 && (header == "hiddenLayerWeights" || header == "hiddenLayerBiases")) {
                hiddenLayerData.push_back(temp);
                temp.clear();
                continue;
            }

            // Skip empty lines
            if (line.length() == 0) continue;

            // Grab and skip header when it shows up
            if (line == "inputWeights") { header = "inputWeights"; continue; }
            else if (line == "hiddenLayerWeights") { header = "hiddenLayerWeights"; continue; }
            else if (line == "hiddenLayerBiases") { header = "hiddenLayerBiases"; continue; }
            else if (line == "outputBiases") { header = "outputBiases"; continue; }

            std::stringstream ss(line);
            std::string value;
            doubleArray_t row;

            while (getline(ss, value, ' ')) {
                row.push_back(std::stod(value));
            }

            // Add data to correct vector, depending on header.
            if (header == "inputWeights") inputWeights.push_back(row);
            else if (header == "hiddenLayerWeights") temp.push_back(row);
            else if (header == "hiddenLayerBiases") temp.push_back(row);
            else if (header == "outputBiases") outputBiases.push_back(row);
        }
    }

    std::vector<matrix> hiddenLayersWeights;
    std::vector<matrix> hiddenLayersBiases;
    for (int i = 0; i < hiddenLayerData.size(); i += 2) {
        hiddenLayersWeights.push_back(hiddenLayerData[i]);
        hiddenLayersBiases.push_back(hiddenLayerData[i + 1]);
    }

    return std::tuple<matrix, matrix, std::vector<matrix>, std::vector<matrix>>(matrix(inputWeights), matrix(outputBiases), hiddenLayersWeights, hiddenLayersBiases);
}

void applyCORSHeaders(auto* res) {
    res->writeHeader("Access-Control-Allow-Origin", "*");
    res->writeHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res->writeHeader("Access-Control-Allow-Headers", "Content-Type");
}

// Parses request body
matrix parseBody(std::string_view body) {
    json j = json::parse(body);
    doubleArray_t data;
    for (auto& x : j.items()) {
        data.push_back(x.value());
    }
    return matrix(data, data.size(), 1);
}

// Parsed predicton given by neural network
int parsePrediction(matrix prediction) {
    int returnVal = -1;
    double confidence = 0.0;
    for (int i = 0; i < prediction.getRows(); i++) {
        if (prediction(i, 0) > confidence) {
            confidence = prediction(i, 0);
            returnVal = i;
        }
    }
    return returnVal;
}

int main() {
    // Initialize model
    MLP model = MLP(784, 10, std::vector<int>{392, 196, 98, 49, 25});

    // Read in weights and biases from text file
    auto [inputWeights, outputBiases, hiddenLayersWeights, hiddenLayersBiases] = readFromFile("../weights/784-392-196-98-49-25-10.txt");
    model.setWeights(inputWeights, hiddenLayersWeights);
    model.setBiases(outputBiases, hiddenLayersBiases);

    // Initialize web-server
    uWS::App().options("/predict", [](auto* res, auto* req) {
        applyCORSHeaders(res);
        res->end("");
        })
        .post("/predict", [&model](auto* res, auto* req) {
            applyCORSHeaders(res);
            res->onData([res, &model](std::string_view body, bool isLast) {
                if (isLast) {
                    auto [hiddenAs, lastA] = model.prediction(parseBody(body));
                    res->end(std::to_string(parsePrediction(lastA)));
                }
                });
            res->onAborted([]() {
                printf("Stream was aborted!\n");
                });
            })
            .listen(PORT, [](auto* listenSocket) {
                if (listenSocket) {
                    std::cout << "Listening to port: " << PORT << std::endl;
                }
                })
                .run();

                std::cout << "Failed to listen on given port, exiting now..." << std::endl;
                return 0;
}