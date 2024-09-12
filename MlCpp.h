#ifndef MY_MATH_LIBRARY_H
#define MY_MATH_LIBRARY_H

#include <Eigen/Dense>
#include <vector>
#include <random>

namespace my_math_library {

// Linear algebra functions
Eigen::MatrixXd multiply_matrices(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

// Statistics functions
double calculate_mean(const Eigen::VectorXd& x);
double calculate_standard_deviation(const Eigen::VectorXd& x);

// Neural network functions
class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs, double learning_rate);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X);

private:
    Eigen::MatrixXd weights1;
    Eigen::MatrixXd weights2;
    Eigen::VectorXd biases1;
    Eigen::VectorXd biases2;

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    Eigen::MatrixXd d_sigmoid(const Eigen::MatrixXd& x);
};

} // namespace my_math_library

#endif