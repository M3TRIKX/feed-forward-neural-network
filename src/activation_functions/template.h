//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_TEMPLATE_H
#define FEEDFORWARDNEURALNET_TEMPLATE_H

/**
 * This class is a template for activation function
 */
class ActivationFunction {
public:
    using type = float;

    /**
     * Computes function at the point x.
     * @param x - point at which we want to compute the function.
     * @return computed value
     */
    static type normal(type x);

    /**
     * Computes function derivative at the point x.
     * @param x - point at which we want to compute the derivative.
     * @return computed derivative at the point x.
     */
    static type derivative(type x);
};

#endif //FEEDFORWARDNEURALNET_TEMPLATE_H
