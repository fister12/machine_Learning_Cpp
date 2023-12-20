#include <iostream>
#include <functional>

class Calculus {
public:
    // Function to calculate the derivative of a function at a given point using finite differences
    static double calculateDerivative(
        const std::function<double(double)>& func,
        double x,
        double epsilon = 1e-6
    ) {
        double xPlusEpsilon = x + epsilon;
        double yPlusEpsilon = func(xPlusEpsilon);

        double derivative = (yPlusEpsilon - func(x)) / epsilon;
        return derivative;
    }

    // Function to perform numerical integration using the trapezoidal rule
    static double integrate(
        const std::function<double(double)>& func,
        double a,
        double b,
        int numIntervals = 1000
    ) {
        double h = (b - a) / numIntervals;
        double result = 0.5 * (func(a) + func(b));

        for (int i = 1; i < numIntervals; ++i) {
            double x = a + i * h;
            result += func(x);
        }

        return result * h;
    }

    // Function to solve a first-order ordinary differential equation (ODE) using Euler's method
    static double solveODE(
        const std::function<double(double, double)>& odeFunc,
        double initialCondition,
        double startTime,
        double endTime,
        double timeStep = 0.1
    ) {
        double currentTime = startTime;
        double currentValue = initialCondition;

        while (currentTime < endTime) {
            double derivative = odeFunc(currentTime, currentValue);
            currentValue += derivative * timeStep;
            currentTime += timeStep;
        }

        return currentValue;
    }
};

int main() {
    // Example usage
    std::function<double(double)> exampleFunction = [](double x) {
        return x * x;
    };

    std::function<double(double, double)> exampleODE = [](double t, double y) {
        // Example first-order ODE: dy/dt = t + y
        return t + y;
    };

    double xValue = 2.0;
    double derivative = Calculus::calculateDerivative(exampleFunction, xValue);
    double integral = Calculus::integrate(exampleFunction, 0.0, 2.0);
    double odeSolution = Calculus::solveODE(exampleODE, 1.0, 0.0, 1.0);

    // Display results
    std::cout << "Derivative at x = " << xValue << ": " << derivative << std::endl;
    std::cout << "Integral from 0 to 2: " << integral << std::endl;
    std::cout << "Solution of dy/dt = t + y with y(0) = 1 at t = 1: " << odeSolution << std::endl;

    return 0;
}
