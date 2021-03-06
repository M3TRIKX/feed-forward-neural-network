//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_TEMPLATE_H
#define FEEDFORWARDNEURALNET_TEMPLATE_H

#include <math.h>
#include "../data_structures/matrix.hpp"

/**
 * This class is a template for activation function
 */
class ActivationFunctionTemplate {
public:
    using type = float;

    /**
     * Computes function at the point x.
     * @param x - point at which we want to compute the function.
     * @return computed value
     */
    static void normal(Matrix<type> &) {};

    /**
     * Computes function derivative at the point x.
     * @param x - point at which we want to compute the derivative.
     * @return computed derivative at the point x.
     */
    static void derivative(Matrix<type> &) {};
};

#endif //FEEDFORWARDNEURALNET_TEMPLATE_H
