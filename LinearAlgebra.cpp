#include <iostream>
#include <vector>

class LinearAlgebra {
public:
    // Vector addition
    static std::vector<double> addVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
        std::vector<double> result;
        if (v1.size() == v2.size()) {
            result.reserve(v1.size());
            for (size_t i = 0; i < v1.size(); ++i) {
                result.push_back(v1[i] + v2[i]);
            }
        }
        return result;
    }

    // Matrix multiplication
    static std::vector<std::vector<double>> multiplyMatrices(
        const std::vector<std::vector<double>>& m1,
        const std::vector<std::vector<double>>& m2
    ) {
        std::vector<std::vector<double>> result;
        if (m1[0].size() == m2.size()) {
            result.resize(m1.size(), std::vector<double>(m2[0].size(), 0.0));

            for (size_t i = 0; i < m1.size(); ++i) {
                for (size_t j = 0; j < m2[0].size(); ++j) {
                    for (size_t k = 0; k < m1[0].size(); ++k) {
                        result[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
        }
        return result;
    }

    // Eigenvalues and eigenvectors (example using diagonal matrix)
    static std::pair<std::vector<double>, std::vector<std::vector<double>>> diagonalizeMatrix(
        const std::vector<std::vector<double>>& matrix
    ) {
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;

        // Assuming 'matrix' is diagonalizable
        // Compute eigenvalues (diagonal elements) and eigenvectors (standard basis vectors)
        for (size_t i = 0; i < matrix.size(); ++i) {
            eigenvalues.push_back(matrix[i][i]);

            std::vector<double> eigenvector(matrix.size(), 0.0);
            eigenvector[i] = 1.0;
            eigenvectors.push_back(eigenvector);
        }

        return {eigenvalues, eigenvectors};
    }
};

int main() {
    // Example usage
    std::vector<double> vector1 = {1.0, 2.0, 3.0};
    std::vector<double> vector2 = {4.0, 5.0, 6.0};

    std::vector<std::vector<double>> matrix1 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {{5.0, 6.0}, {7.0, 8.0}};

    std::vector<double> resultVector = LinearAlgebra::addVectors(vector1, vector2);
    std::vector<std::vector<double>> resultMatrix = LinearAlgebra::multiplyMatrices(matrix1, matrix2);

    // Display results
    std::cout << "Vector Addition Result: ";
    for (const auto& elem : resultVector) {
        std::cout << elem << " ";
    }
    std::cout << "\n\nMatrix Multiplication Result:\n";
    for (const auto& row : resultMatrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
