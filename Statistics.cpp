#include <iostream>
#include <vector>
#include <cmath>

class Statistics {
public:
    // Function to calculate the mean of a dataset
    static double calculateMean(const std::vector<double>& data) {
        if (data.empty()) {
            std::cerr << "Error: Empty dataset\n";
            return 0.0;
        }

        double sum = 0.0;
        for (const auto& value : data) {
            sum += value;
        }

        return sum / data.size();
    }

    // Function to calculate the variance of a dataset
    static double calculateVariance(const std::vector<double>& data) {
        if (data.size() < 2) {
            std::cerr << "Error: Variance calculation requires at least two data points\n";
            return 0.0;
        }

        double mean = calculateMean(data);
        double sumSquaredDeviations = 0.0;

        for (const auto& value : data) {
            double deviation = value - mean;
            sumSquaredDeviations += deviation * deviation;
        }

        return sumSquaredDeviations / (data.size() - 1);
    }

    // Function to calculate the standard deviation of a dataset
    static double calculateStandardDeviation(const std::vector<double>& data) {
        return std::sqrt(calculateVariance(data));
    }

    // Function to perform a two-sample t-test for illustration purposes
    static bool performTTest(
        const std::vector<double>& sample1,
        const std::vector<double>& sample2
    ) {
        // Assuming both samples have equal variance
        double mean1 = calculateMean(sample1);
        double mean2 = calculateMean(sample2);

        double variance1 = calculateVariance(sample1);
        double variance2 = calculateVariance(sample2);

        double standardError = std::sqrt(variance1 / sample1.size() + variance2 / sample2.size());

        // Calculate t-statistic
        double tStat = std::abs(mean1 - mean2) / standardError;

        // Perform a two-tailed t-test with significance level 0.05
        const double criticalValue = 2.262;  // For a two-tailed test at 0.05 significance level and degrees of freedom = total_sample_size - 2

        return tStat > criticalValue;
    }
};

int main() {
    // Example usage
    std::vector<double> dataset1 = {25.4, 27.8, 23.5, 26.1, 24.9};
    std::vector<double> dataset2 = {22.1, 20.8, 25.2, 21.7, 23.5};

    double mean1 = Statistics::calculateMean(dataset1);
    double variance1 = Statistics::calculateVariance(dataset1);
    double stdDev1 = Statistics::calculateStandardDeviation(dataset1);

    double mean2 = Statistics::calculateMean(dataset2);
    double variance2 = Statistics::calculateVariance(dataset2);
    double stdDev2 = Statistics::calculateStandardDeviation(dataset2);

    bool tTestResult = Statistics::performTTest(dataset1, dataset2);

    // Display results
    std::cout << "Dataset 1: Mean = " << mean1 << ", Variance = " << variance1 << ", Standard Deviation = " << stdDev1 << "\n";
    std::cout << "Dataset 2: Mean = " << mean2 << ", Variance = " << variance2 << ", Standard Deviation = " << stdDev2 << "\n";
    std::cout << "Two-sample t-test result: " << (tTestResult ? "Reject null hypothesis" : "Fail to reject null hypothesis") << "\n";

    return 0;
}
