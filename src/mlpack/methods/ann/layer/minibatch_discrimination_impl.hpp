/**
 * @file methods/ann/layer/minibatch_discrimination_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the MiniBatchDiscrimination layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_IMPL_HPP

// In case it hasn't yet been included.
#include "minibatch_discrimination.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
MiniBatchDiscriminationType<InputType, OutputType
>::MiniBatchDiscriminationType() :
  A(0),
  B(0),
  C(0),
  batchSize(0)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
MiniBatchDiscriminationType<InputType, OutputType
>::MiniBatchDiscriminationType(
    const size_t inSize,
    const size_t outSize,
    const size_t features) :
    A(inSize),
    B(outSize - inSize),
    C(features),
    batchSize(0)
{
  weights.set_size(A * B * C, 1);
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Reset()
{
  weight = InputType(weights.memptr(), B * C, A, false, false); 
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  tempM = weight * input;
  M = arma::cube(tempM.memptr(), B, C, batchSize, false, false);
  distances.set_size(B, batchSize, batchSize);
  output.set_size(B, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    output.col(i).ones();
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (j < i)
      {
        output.col(i) += distances.slice(j).col(i);
      }
      else if (i == j)
      {
        continue;
      }
      else
      {
        distances.slice(i).col(j) =
          arma::exp(-arma::sum(abs(M.slice(i) - M.slice(j)), 1));
        output.col(i) += distances.slice(i).col(j);
      }
    }
  }

  output = join_cols(input, output); // (A + B) x batchSize
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = gy.head_rows(A);
  InputType gM = gy.tail_rows(B);
  deltaM.zeros(B, C, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (i == j)
      {
        continue;
      }
      InputType t = arma::sign(M.slice(i) - M.slice(j));
      t.each_col() %=
          distances.slice(std::min(i, j)).col(std::max(i, j)) % gM.col(i);
      deltaM.slice(i) -= t;
      deltaM.slice(j) += t;
    }
  }

  deltaTemp = InputType(deltaM.memptr(), B * C, batchSize, false, false); 
  g += weight.t() * deltaTemp;
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& /* error */,
    OutputType& gradient)
{
  gradient = arma::vectorise(deltaTemp * input.t());
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MiniBatchDiscriminationType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(A));
  ar(CEREAL_NVP(B));
  ar(CEREAL_NVP(C));

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (cereal::is_loading<Archive>())
  {
    weights.set_size(A * B * C, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif/**
 * @file methods/ann/layer/minibatch_discrimination_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the MiniBatchDiscrimination layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_IMPL_HPP

// In case it hasn't yet been included.
#include "minibatch_discrimination.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
MiniBatchDiscriminationType<InputType, OutputType
>::MiniBatchDiscriminationType() :
  A(0),
  B(0),
  C(0),
  batchSize(0)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
MiniBatchDiscriminationType<InputType, OutputType
>::MiniBatchDiscriminationType(
    const size_t inSize,
    const size_t outSize,
    const size_t features) :
    A(inSize),
    B(outSize - inSize),
    C(features),
    batchSize(0)
{
  weights.set_size(A * B * C, 1);
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Reset()
{
  weight = InputType(weights.memptr(), B * C, A, false, false); 
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  tempM = weight * input;
  M = arma::cube(tempM.memptr(), B, C, batchSize, false, false);
  distances.set_size(B, batchSize, batchSize);
  output.set_size(B, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    output.col(i).ones();
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (j < i)
      {
        output.col(i) += distances.slice(j).col(i);
      }
      else if (i == j)
      {
        continue;
      }
      else
      {
        distances.slice(i).col(j) =
          arma::exp(-arma::sum(abs(M.slice(i) - M.slice(j)), 1));
        output.col(i) += distances.slice(i).col(j);
      }
    }
  }

  output = join_cols(input, output); // (A + B) x batchSize
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = gy.head_rows(A);
  InputType gM = gy.tail_rows(B);
  deltaM.zeros(B, C, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (i == j)
      {
        continue;
      }
      InputType t = arma::sign(M.slice(i) - M.slice(j));
      t.each_col() %=
          distances.slice(std::min(i, j)).col(std::max(i, j)) % gM.col(i);
      deltaM.slice(i) -= t;
      deltaM.slice(j) += t;
    }
  }

  deltaTemp = InputType(deltaM.memptr(), B * C, batchSize, false, false); 
  g += weight.t() * deltaTemp;
}

template<typename InputType, typename OutputType>
void MiniBatchDiscriminationType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& /* error */,
    OutputType& gradient)
{
  gradient = arma::vectorise(deltaTemp * input.t());
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MiniBatchDiscriminationType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(A));
  ar(CEREAL_NVP(B));
  ar(CEREAL_NVP(C));

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (cereal::is_loading<Archive>())
  {
    weights.set_size(A * B * C, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
