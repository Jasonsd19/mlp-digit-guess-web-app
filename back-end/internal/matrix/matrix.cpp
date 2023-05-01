#include "matrix.h"
#include <thread>
#include <math.h>
#include <algorithm>

matrix::matrix() {
    this->rows = 0;
    this->columns = 0;
}

matrix::matrix(doubleArray_t data, int rows, int columns) {
    this->rows = rows;
    this->columns = columns;

    if (data.size() == rows * columns) {
        mData = data;
    }
    else {
        data.resize(rows * columns);
        mData = data;
    }
}

matrix::matrix(doubleArray_t data, int rowsColumns) {
    this->rows = rowsColumns;
    this->columns = rowsColumns;

    if (data.size() == rows * columns) {
        mData = data;
    }
    else {
        data.resize(rows * columns);
        mData = data;
    }
}

matrix::matrix(twoDimDoubleArray_t data) {
    this->rows = data.size();
    this->columns = data[0].size();

    mData = doubleArray_t(rows * columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            mData[columns * i + j] = data[i][j];
        }
    }
}

matrix::~matrix() {

}

doubleArray_t matrix::getData() {
    return mData;
}

unsigned int matrix::getRows() {
    return rows;
}

unsigned int matrix::getColumns() {
    return columns;
}

// Checks if two matrices have the same number of rows and columns
bool matrix::sameDims(matrix M1, matrix M2) {
    return M1.getColumns() == M2.getColumns() && M1.getRows() == M2.getRows();
}

// Piece-wise addition of two matrices
matrix matrix::operator+(matrix m) {
    if (!sameDims((*this), m)) {
        throw std::logic_error("Mismatched dimensions between matrices.");
    }

    doubleArray_t operandData = m.getData();
    doubleArray_t newData = doubleArray_t(rows * columns);

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    newData[columns * i + j] = mData[columns * i + j] + operandData[columns * i + j];
                }
            }
        }
    }

    return matrix(newData, rows, columns);
}

// Piece-wise subtraction of two matrices
matrix matrix::operator-(matrix m) {
    if (!sameDims((*this), m)) {
        throw std::logic_error("Mismatched dimensions between matrices.");
    }

    doubleArray_t operandData = m.getData();
    doubleArray_t newData = doubleArray_t(rows * columns);

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    newData[columns * i + j] = mData[columns * i + j] - operandData[columns * i + j];
                }
            }
        }
    }

    return matrix(newData, rows, columns);
}

// Piece-wise multiplication of two matrices
matrix matrix::operator*(matrix m) {
    if (!sameDims((*this), m)) {
        throw std::logic_error("Mismatched dimensions between matrices.");
    }

    doubleArray_t operandData = m.getData();
    doubleArray_t newData = doubleArray_t(rows * columns);

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    newData[columns * i + j] = mData[columns * i + j] * operandData[columns * i + j];
                }
            }
        }
    }

    return matrix(newData, rows, columns);
}

// Piece-wise division of two matrices
matrix matrix::operator/(matrix m) {
    if (!sameDims((*this), m)) {
        throw std::logic_error("Mismatched dimensions between matrices.");
    }

    doubleArray_t operandData = m.getData();
    doubleArray_t newData = doubleArray_t(rows * columns);

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    newData[columns * i + j] = mData[columns * i + j] / operandData[columns * i + j];
                }
            }
        }
    }

    return matrix(newData, rows, columns);
}

matrix matrix::operator=(matrix m) {
    mData = m.getData();
    rows = m.getRows();
    columns = m.getColumns();
    return *this;
}

// Return the matrix element at the specified position (row, column)
double matrix::operator()(unsigned int i, unsigned int j) {
    if (i * j > mData.size() - 1) {
        throw std::out_of_range("Index is out of range");
    }

    return mData[columns * i + j];
}

// Prints the first 10 rows and columns to standard output
std::ostream& operator<<(std::ostream& os, matrix& M) {
    int maxDim = 10;
    int rows = std::min(maxDim, (int)M.rows);
    int columns = std::min(maxDim, (int)M.columns);
    doubleArray_t* matrixData = &M.mData;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++)
        {
            os << (*matrixData)[M.columns * i + j] << " ";
        }
        os << std::endl;
    }
    return os;
}

// Returns the specified row
matrix matrix::getRow(matrix M, int row) {
    if (row >= M.rows) {
        throw std::logic_error("Row index out of range");
    }

    std::vector<double> rowData;
    for (int i = 0; i < M.columns; i++) {
        rowData.push_back(M(row, i));
    }
    return matrix(rowData, 1, M.columns);
}

// Returns the specified column
matrix matrix::getColumn(matrix M, int column) {
    if (column >= M.columns) {
        throw std::logic_error("Column index out of range");
    }

    std::vector<double> columnData;
    for (int i = 0; i < M.rows; i++) {
        columnData.push_back(M(i, column));
    }
    return matrix(columnData, M.rows, 1);
}

// Returns the input matrix with its elements multiplied by a scalar
matrix matrix::scalarMultiply(matrix M, double scalar) {
    int rows = M.rows;
    int columns = M.columns;

    doubleArray_t newData = doubleArray_t(rows * columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            newData[columns * i + j] = scalar * M(i, j);
        }
    }
    return matrix(newData, rows, columns);
}

// Returns the product of post-multiplication of the left matrix by the right matrix. Columns of the left
// matrix must match the rows of the right matrix.
matrix matrix::matrixMultiply(matrix leftMatrix, matrix rightMatrix) {
    if (leftMatrix.columns != rightMatrix.rows) {
        throw std::logic_error("The number of columns of the left matrix must equal the number of rows of the right matrix.");
    }

    const int MAX_THREADS = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int rows = leftMatrix.rows;
    int columns = leftMatrix.columns;
    int rmColumns = rightMatrix.columns;

    doubleArray_t newData = doubleArray_t(rows * rmColumns);

    // Define block size so we can use matrix blocking (this is specific to my CPU and architecture).
    // I tested various block sizes 8, 16, 32, 64, 128, 256, ..., and 64 seemed to perform the best.

    auto matMulLoop = [&](int loopStart, int loopStep) {
        // This outer loop sets up matrix blocking in one dimension. I found blocking in one dimension performed
        // better than blocking in 2 or 3 dimensions.
        for (int kk = 0; kk < columns; kk += blockSizeMultiply) {
            int kkMin = std::min(kk + blockSizeMultiply, columns);
            // Change loop order to maximize cache locality. Naive implementation is looping through i,j,k 
            // where i is the # of rows of the left matrix, j is the # of columns of the right matrix, and k is the
            // the number of multiplications between a row of the left matrix and column of the right matrix. Changing
            // the loop order to i,k,j makes it more cache friendly, resulting in less cache misses and increased performance.
            for (int i = loopStart; i < loopStart + loopStep; i++) {
                for (int k = kk; k < kkMin; k++) {
                    for (int j = 0; j < rmColumns; j++) {
                        newData[rmColumns * i + j] += leftMatrix(i, k) * rightMatrix(k, j);
                    }
                }
            }
        }
    };

    // Check if the matrices are larger than 300x300 (approximately). If they are then run
    // the matrix multiplication with threading. Otherwise run w/o threading. The overhead 
    // of initializing the threads makes using threading for small matrices inefficient.
    if (rows + columns + rmColumns >= 900) {
        // Divide matrix rows into equal sizes for each thread
        int intDiv = rows / MAX_THREADS;
        int remainder = rows % MAX_THREADS;
        // Keeps track of where each matrix multiplication loop starts
        int loopStart = 0;
        for (int i = 0; i < MAX_THREADS; i++) {
            if (i < remainder) {
                threads.emplace_back(matMulLoop, loopStart, intDiv + 1);
                loopStart += intDiv + 1;
            }
            else {
                threads.emplace_back(matMulLoop, loopStart, intDiv);
                loopStart += intDiv;
            }
        }

        for (int i = 0; i < MAX_THREADS; i++) {
            threads[i].join();
        }
    }
    else {
        matMulLoop(0, rows);
    }

    return matrix(newData, rows, rmColumns);
}

// Returns the transpose of the input matrix.
matrix matrix::transpose(matrix M) {
    const int MAX_THREADS = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int rows = M.rows;
    int columns = M.columns;

    doubleArray_t newData = doubleArray_t(rows * columns);

    // Define block size so we can use matrix blocking (this is specific to my CPU and architecture).
    // I tested various block sizes 8, 16, 32, 64, 128, 256, ..., and 8 seemed to perform the best.

    auto matTransposeLoop = [&](int loopStart, int loopStep) {
        for (int ii = loopStart; ii < loopStep; ii += blockSizeTranspose) {
            int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
            for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
                int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

                for (int i = ii; i < iiMin; i++) {
                    for (int j = jj; j < jjMin; j++) {
                        newData[rows * j + i] = M(i, j);
                    }
                }
            }
        }
    };

    // Check if the matrices are larger than 1024x1024 (approximately). If they are then run
    // the matrix transpose with threading. Otherwise run w/o threading. The overhead 
    // of initializing the threads makes using threading for small matrices inefficient.
    if (rows + columns >= 2048) {
        // Divide matrix rows into equal sizes for each thread
        int intDiv = rows / MAX_THREADS;
        int remainder = rows % MAX_THREADS;
        // Keeps track of where each matrix multiplication loop starts
        int loopStart = 0;
        for (int i = 0; i < MAX_THREADS; i++) {
            if (i < remainder) {
                threads.emplace_back(matTransposeLoop, loopStart, intDiv + 1);
                loopStart += intDiv + 1;
            }
            else {
                threads.emplace_back(matTransposeLoop, loopStart, intDiv);
                loopStart += intDiv;
            }
        }

        for (int i = 0; i < MAX_THREADS; i++) {
            threads[i].join();
        }
    }
    else {
        matTransposeLoop(0, rows);
    }

    return matrix(newData, columns, rows);
}

// Returns a n x n identity matrix.
matrix matrix::identityMatrix(int n) {
    doubleArray_t newData = doubleArray_t(n * n);

    for (int i = 0; i < n; i++) {
        newData[n * i + i] = 1.0;
    }

    return matrix(newData, n);
}

// Returns the Permutation matrix, Upper Triangular matrix, Lower Triangular matrix, and number of row swaps,
// that result from performing LUP factorization/decomposition on the input matrix. This function only accepts square matrices.
std::tuple<matrix, matrix, matrix, int> matrix::LUPDecompose(matrix M) {
    if (M.rows != M.columns) {
        throw std::logic_error("The matrix must be square.");
    }

    int n = M.rows;
    int swaps = 0;

    // permutations matrix data, we will use the extra index to store the 
    // number of row swaps which is needed when calculating the determinant
    doubleArray_t pData = doubleArray_t(n * n);
    // lower triangular matrix data
    doubleArray_t lData = doubleArray_t(n * n);

    // Initialize p-matrix and l-matrix data as identity matrices
    for (int i = 0; i < n; i++) {
        pData[n * i + i] = 1.0;
        lData[n * i + i] = 1.0;
    }

    // upper triangular matrix data (which we initialize as a copy of the input matrix data)
    doubleArray_t uData = M.getData();

    // Perform partial pivoting
    for (int i = 0; i < n; i++) {

        double largestVal = 0.0;
        int largestValIndex = 0;
        // Iterate through each column and find the largest absolute value
        for (int j = i; j < n; j++) {
            double absVal = std::abs(uData[n * j + i]);
            if (absVal > largestVal) {
                largestVal = absVal;
                largestValIndex = n * j + i;
            }
        }

        // If the largest value in the column is not in the pivot position, then swap rows so it is.
        if (largestValIndex != n * i + i) {
            int largestValRow = largestValIndex / n;

            // Swap rows of upper matrix (which is a copy of input matrix) and permutation matrix
            for (int j = 0; j < n; j++) {
                double inputTemp = uData[n * i + j];
                uData[n * i + j] = uData[n * largestValRow + j];
                uData[n * largestValRow + j] = inputTemp;

                double permutationTemp = pData[n * i + j];
                pData[n * i + j] = pData[n * largestValRow + j];
                pData[n * largestValRow + j] = permutationTemp;

            }
            swaps++;
        }
    }

    // Perform PLU decomposition
    for (int i = 0; i < n; i++) {
        // Iterate through all (n - i + 1) rows
        for (int j = i + 1; j < n; j++) {
            // Multiplier is used to multiply each value in row j and is used in the lower matrix
            double multiplier = uData[n * j + i] / uData[(n * i + i)];
            // Iterate through all columns in row j
            for (int k = i; k < n; k++) {
                uData[n * j + k] -= multiplier * uData[(n * i) + k];
            }
            // Record multiplier in the lower matrix
            lData[n * j + i] = multiplier;
        }
    }

    // Create all return matrices with the above calculated data
    matrix pMatrix = matrix(pData, n);
    matrix uMatrix = matrix(uData, n);
    matrix lMatrix = matrix(lData, n);

    return std::tuple<matrix, matrix, matrix, int>(pMatrix, uMatrix, lMatrix, swaps);
}

// Solves Ax = b and returns the solution vector (x) to the input matrix(A) and vector (b).
// Input matrix M must be square. Will only return unique solutions, throws error on non-unique solutions.
matrix matrix::solve(matrix M, matrix vector) {
    auto [pMatrix, uMatrix, lMatrix, swaps] = LUPDecompose(M);
    return solveLUP(lMatrix, uMatrix, pMatrix, swaps, vector);
}

// Takes in previously calculated Lower Triangular Matrix, Upper Triangular Matrix, Permutation Matrix, and swaps.
// Matrices L, U, and P must be square and have the same dimensions.
// Solves Ax=b and returns the solution vector (x) to the input matrix(PA = LU) and vector (b).
// Will only return unique solutions, throws error on non-unique solutions.
matrix matrix::solveLUP(matrix L, matrix U, matrix P, int swaps, matrix b) {
    // Use threshold to check if value is near enough to zero.
    double threshold = std::numeric_limits<double>::epsilon();
    // If the determinant is (near) zero, then there is no unique solution.
    double determinant = determinantLUP(L, U, swaps);
    if (determinant < threshold && determinant > -threshold) {
        throw std::logic_error("There is no unique solution.");
    }

    int n = L.rows;

    // First solve Ly = Pb for y
    doubleArray_t yData = doubleArray_t(n);

    // Pb is a n*1 vector
    matrix Pb = matrixMultiply(P, b);

    // Iterate through each row of the lower matrix
    for (int i = 0; i < n; i++) {
        double result = 0.0;
        // Iterate through each column in the row upto (and including) the diagonal element
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                result += Pb(i, 0) / L(i, j);
            }
            else {
                result -= L(i, j) * yData[j];
            }
        }
        yData[i] = result;
    }

    // Second we solve Ux = y for x
    doubleArray_t xData = doubleArray_t(n);

    // Iterate through each row of the upper matrix
    for (int i = n - 1; i >= 0; i--) {
        double result = 0.0;
        // Iterate through each column starting at the diagonal
        for (int j = i; j < n; j++) {
            if (i == j) {
                result += yData[i] / U(i, j);
            }
            else {
                result -= (U(i, j) / U(i, i)) * xData[j];
            }
        }
        xData[i] = result;
    }

    return matrix(xData, n, 1);
}

// Returns the determinant of the input matrix. Input matrix must be square.
double matrix::determinant(matrix M) {
    auto [_, uMatrix, lMatrix, swaps] = LUPDecompose(M);
    return determinantLUP(lMatrix, uMatrix, swaps);
}

// Takes in previously calculated Lower Triangular Matrix, Upper Triangular Matrix, and swaps.
// Returns the determinant of the input matrix (PA = LU). Input matrices must be square and have same dimensions.
double matrix::determinantLUP(matrix L, matrix U, int swaps) {
    if (L.columns != L.rows || U.columns != U.rows) {
        throw std::logic_error("One or more of the matrices provided are not square.");
    }

    if (!sameDims(L, U)) {
        throw std::logic_error("Lower, and Upper matrices must have the same dimensions.");
    }

    int n = L.rows;
    double result = 1.0;

    for (int i = 0; i < n; i++) {
        result *= L(i, i) * U(i, i);
    }

    return swaps % 2 == 0 ? result : -result;
}

// Returns the inverse of the input matrix. Input matrix must be square.
matrix matrix::inverse(matrix M) {
    auto [pMatrix, uMatrix, lMatrix, swaps] = LUPDecompose(M);
    return inverseLUP(lMatrix, uMatrix, pMatrix, swaps);
}

// Takes in previously calculated Lower Triangular Matrix, Upper Triangular Matrix, Permutation Matrix, and swaps.
// Returns the inverse of the input matrix by solving (A)(A^-1) = I. Input matrices must be square and have same dimensions.
matrix matrix::inverseLUP(matrix L, matrix U, matrix P, int swaps) {
    if (L.rows != L.columns || U.rows != U.columns || P.rows != P.columns) {
        throw std::logic_error("One or more of the matrices provided are not square.");
    }

    if (!sameDims(L, U) || !sameDims(U, P) || !sameDims(L, P)) {
        throw std::logic_error("Lower, Upper, and Permutation matrices must have the same dimensions.");
    }

    int n = L.rows;
    doubleArray_t inverseMatData = doubleArray_t(n * n);

    for (int i = 0; i < n; i++) {
        doubleArray_t identityCol = doubleArray_t(n);
        identityCol[i] = 1.0;
        doubleArray_t inverseCol = solveLUP(L, U, P, swaps, matrix(identityCol, n, 1)).mData;

        for (int j = 0; j < n; j++) {
            inverseMatData[L.columns * j + i] = inverseCol[j];
        }
    }

    return matrix(inverseMatData, n);
}

// Sums all elements together and returns the result.
double matrix::sum(matrix M) {
    int rows = M.rows;
    int columns = M.columns;
    double result = 0.0;

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    result += M(i, j);
                }
            }
        }
    }

    return result;
}


// Apply the given function to each element in the input matrix.
matrix matrix::map(matrix M, double (*f)(double)) {
    int rows = M.rows;
    int columns = M.columns;
    doubleArray_t mappedValues;

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    mappedValues.push_back(f(M(i, j)));
                }
            }
        }
    }

    return matrix(mappedValues, M.rows, M.columns);
}

// Apply the given function to each element in the input matrix.
// Then sum all elements together and return the result.
double matrix::mapSum(matrix M, double (*f)(double)) {
    int rows = M.rows;
    int columns = M.columns;
    double result = 0.0;

    for (int ii = 0; ii < rows; ii += blockSizeTranspose) {
        int iiMin = std::min(ii + blockSizeTranspose, (int)rows);
        for (int jj = 0; jj < columns; jj += blockSizeTranspose) {
            int jjMin = std::min(jj + blockSizeTranspose, (int)columns);

            for (int i = ii; i < iiMin; i++) {
                for (int j = jj; j < jjMin; j++) {
                    result += f(M(i, j));
                }
            }
        }
    }

    return result;
}