#include <sstream>
#include <string>
#include <fstream>
#include <matrix.h>

// Specifically implemented to pull in and shape data from the MNIST dataset.
namespace csv {
    // Reads in a csv file containing labeled training data. The first column contains the labels.
    // Returns a column vector of labels and a matrix of data, where each row is a training example.
    std::tuple<matrix, matrix> read_data(const std::string& filename, int outputLayerSize) {
        twoDimDoubleArray_t data;
        doubleArray_t labels;

        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;

            // Skip header
            getline(file, line);

            while (getline(file, line)) {
                int i = 0;
                std::stringstream ss(line);
                std::string value;
                std::vector<double> row;

                while (getline(ss, value, ',')) {
                    i == 0 ? labels.push_back(std::stod(value)) : row.push_back(std::stod(value));
                    i++;
                }

                if (row.size() > 0) data.push_back(row);
            }
        }

        // Modify labels to conform to the MLP models output layer.
        twoDimDoubleArray_t modLabelData;
        for (int i = 0; i < labels.size(); i++) {
            doubleArray_t modLabel;
            for (int j = 0; j < outputLayerSize; j++) {
                j == (int)labels[i] ? modLabel.push_back(1.0) : modLabel.push_back(0.0);
            }
            modLabelData.push_back(modLabel);
        }

        return std::tuple<matrix, matrix>(matrix(data), matrix(modLabelData));
    }
}