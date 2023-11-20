#include <iostream>
#include <vector>
#include <cmath>

// Simple linear regression class
class LinearRegression {
private:
    double slope;
    double intercept;

public:
    LinearRegression() : slope(0), intercept(0) {}

    // Fit the model to the training data
    void fit(const std::vector<double>& X, const std::vector<double>& y) {
        // Calculate mean of X and y
        double meanX = 0, meanY = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            meanX += X[i];
            meanY += y[i];
        }
        meanX /= X.size();
        meanY /= y.size();

        // Calculate slope and intercept
        double numerator = 0, denominator = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            numerator += (X[i] - meanX) * (y[i] - meanY);
            denominator += std::pow((X[i] - meanX), 2);
        }

        slope = numerator / denominator;
        intercept = meanY - slope * meanX;
    }

    // Predict the output for new data points
    double predict(double x) const {
        return slope * x + intercept;
    }

    // Get the slope and intercept
    double getSlope() const {
        return slope;
    }

    double getIntercept() const {
        return intercept;
    }
};

int main() {
    // Example usage
    std::vector<double> X = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.5, 3.5, 4.5, 5.5, 6.5};

    LinearRegression lr;
    lr.fit(X, y);

    std::cout << "Coefficients: Slope = " << lr.getSlope() << ", Intercept = " << lr.getIntercept() << std::endl;

    // Make predictions
    double newX = 6.0;
    double prediction = lr.predict(newX);
    std::cout << "Prediction for X = " << newX << ": " << prediction << std::endl;

    return 0;
}
