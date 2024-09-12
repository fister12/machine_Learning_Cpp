#include "MlCpp.h"

namespace my_math_library {

Eigen::MatrixXd multiply_matrices(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    return A * B;
}

Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    return A.colPivHouseholderQr().solve(b);
}

double calculate_mean(const Eigen::VectorXd& x) {
    return x.sum() / x.size();
}

double calculate_standard_deviation(const Eigen::VectorXd& x) {
    double mean = calculate_mean(x);
    return sqrt((x.array() - mean).square().sum() / (x.size() - 1));
}

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size) {
    // Initialize weights and biases randomly
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    weights1 = Eigen::MatrixXd::Random(hidden_size, input_size);
    weights2 = Eigen::MatrixXd::Random(output_size, hidden_size);
    biases1 = Eigen::VectorXd::Random(hidden_size);
    biases2 = Eigen::VectorXd::Random(output_size);
}

void NeuralNetwork::train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Eigen::MatrixXd z1 = weights1 * X.transpose() + biases1.replicate(1, X.rows());
        Eigen::MatrixXd a1 = sigmoid(z1);
        Eigen::MatrixXd z2 = weights2 * a1 + biases2.replicate(1, X.rows());
        Eigen::MatrixXd a2 = sigmoid(z2);

        // Backpropagation
        Eigen::MatrixXd d_a2 = a2 - y.transpose();
        Eigen::MatrixXd d_z2 = d_a2.array() * d_sigmoid(z2).array();
        Eigen::MatrixXd d_weights2 = d_z2 * a1.transpose();
        Eigen::MatrixXd d_biases2 = d_z2.colwise().sum();

        Eigen::MatrixXd d_a1 = weights2.transpose() * d_z2;
        Eigen::MatrixXd d_z1 = d_a1.array() * d_sigmoid(z1).array();
        Eigen::MatrixXd d_weights1 = d_z1 * X;
        Eigen::MatrixXd d_biases1 = d_z1.colwise().sum();

        // Update weights and biases
        weights1 -= learning_rate * d_weights1;
        weights2 -= learning_rate * d_weights2;
        biases1 -= learning_rate * d_biases1;
        biases2 -= learning_rate * d_biases2;
    }
}

Eigen::MatrixXd NeuralNetwork::predict(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd z1 = weights1 * X.transpose() + biases1.replicate(1, X.rows());
    Eigen::MatrixXd a1 = sigmoid(z1);
    Eigen::MatrixXd z2 = weights2 * a1 + biases2.replicate(1, X.rows());
    Eigen::MatrixXd a2 = sigmoid(z2);
    return a2.transpose();
}

Eigen::MatrixXd NeuralNetwork::sigmoid(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

Eigen::MatrixXd NeuralNetwork::d_sigmoid(const Eigen::MatrixXd& x) {
    return sigmoid(x).array() * (1.0 - sigmoid(x).array());
}

} 