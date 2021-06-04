/**
 * @file methods/ann/layer/instance_norm_impl.hpp
 * @author Anjishnu Mukherjee
 * @author Shah Anwaar Khalid
 *
 * Implementation of the Instance Normalization Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INSTANCE_NORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_INSTANCE_NORM_IMPL_HPP

// In case it is not included.
#include "instance_norm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
InstanceNorm<InputDataType, OutputDataType>::InstanceNorm() :
    size(0),
    eps(1e-8),
    average(true),
    affine(true),
    momentum(0.0),
    deterministic(false),
    reset(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
InstanceNorm<InputDataType, OutputDataType>::InstanceNorm(
    const size_t size,
    const size_t batchSize,
    const double eps,
    const bool average,
    const bool affine,
    const double momentum) :
    size(size),
    batchSize(batchSize),
    eps(eps),
    average(average),
    affine(affine),
    momentum(momentum),
    deterministic(false),
    reset(false)
{
    batchNorm = ann::BatchNorm<> (size * batchSize,
                                  eps,
                                  average,
                                  affine,
                                  momentum);
    runningMean.zeros(size, 1);
    runningVariance.ones(size, 1);
    runningVariance = batchNorm.TrainingVariance();
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void InstanceNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  // Instance Norm with (N, C, H, W) is same as Batch Norm with (1, N*C, H, W),
  // where N is the batchSize, C is the number of channels, H and W are the
  // height and width of each image respectively.
  if (input.n_cols != batchSize)
  {
    Log::Fatal<<"Must use the same BatchSize that was used in the constructor"
                <<std::endl;
  }
  if (!reset)
  {
    batchNorm.Reset();
    reset = true;
  }

  shapeA = input.n_rows;
  shapeB = input.n_cols;

  if (deterministic)
    batchNorm.Deterministic() = true;

  arma::mat inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(), shapeA*shapeB,
        1, false, false);
  batchNorm.Forward(inputTemp, output);
  output.reshape(shapeA, shapeB);
  runningMean = batchNorm.TrainingMean();
  runningMean.reshape(size, shapeB);
  runningMean = arma::mean(runningMean, 1);
  runningVariance = batchNorm.TrainingVariance();
  runningVariance.reshape(size, shapeB);
  runningVariance = arma::mean(runningVariance, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void InstanceNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  arma::mat inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(), shapeA*shapeB,
      1, false, false);
  arma::mat gyTemp(const_cast<arma::Mat<eT>&>(gy).memptr(), shapeA*shapeB,
      1, false, false);
  batchNorm.Backward(inputTemp, gyTemp, g);
  g.reshape(shapeA, shapeB);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void InstanceNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  arma::mat inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(), shapeA*shapeB,
      1, false, false);
  arma::mat errorTemp(const_cast<arma::Mat<eT>&>(error).memptr(), shapeA*shapeB,
      1, false, false);
  batchNorm.Gradient(inputTemp, errorTemp, gradient);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void InstanceNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(average));
  ar(CEREAL_NVP(affine));
  ar(CEREAL_NVP(momentum));
  ar(CEREAL_NVP(deterministic));
  ar(CEREAL_NVP(runningMean));
  ar(CEREAL_NVP(runningVariance));
  ar(CEREAL_NVP(reset));
  ar(CEREAL_NVP(shapeA));
  ar(CEREAL_NVP(shapeB));
  ar(CEREAL_NVP(batchNorm));
}

} // namespace ann
} // namespace mlpack

#endif
