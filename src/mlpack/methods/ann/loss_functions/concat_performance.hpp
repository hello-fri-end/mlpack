/**
 * @file methods/ann/layer/concat_performance.hpp
 * @author Marcus Edel
 *
 * Definition of the ConcatPerformance class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_HPP

#include <mlpack/prereqs.hpp>
#include<mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>


namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the concat performance class. The class works as a
 * feed-forward fully connected network container which plugs performance layers
 * together.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename OutputLayerType = NegativeLogLikelihood<>,  
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class ConcatPerformance:
{
 public:
  /**
   * Create the ConcatPerformance object.
   *
   * @param inSize The number of inputs.
   * @param outputLayer Output layer used to evaluate the network.
   */
  ConcatPerformance(const size_t inSize = 1,
                    OutputLayerType&& outputLayer = OutputLayerType());

  /*
   * Computes the Negative log likelihood.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  double Forward(const InputType& input, OutputType& target);

  /**
   * Ordinary feed backward pass of a neural network. The negative log
   * likelihood layer expectes that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  void Backward(const InputType& input,
                const OutputType& target,
                OutputType& output);

  //! Get the output parameter.
  OutputType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the number of inputs.
  size_t InSize() const { return inSize; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */);

 private:
  //! Locally-stored number of inputs.
  size_t inSize;

  //! Instantiated outputlayer used to evaluate the network.
  OutputLayerType outputLayer;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
}; // class ConcatPerformance
typedef ConcatPerformance <arma::mat, arma::mat> ConcatPerformance;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_performance_impl.hpp"

#endif
