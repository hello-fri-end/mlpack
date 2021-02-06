/**
 * @file methods/ann/gan/gan_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_GAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_GAN_IMPL_HPP

#include "gan.hpp"

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/network_init.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network.  */ {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::GAN(
    Model generator,
    Model discriminator,
    InitializationRuleType& initializeRule,
    Noise& noiseFunction,
    const size_t noiseDim,
    const size_t batchSize,
    const size_t generatorUpdateStep,
    const size_t preTrainSize,
    const double multiplier,
    const double clippingParameter,
    const double lambda):
    generator(std::move(generator)),
    discriminator(std::move(discriminator)),
    initializeRule(initializeRule),
    noiseFunction(noiseFunction),
    noiseDim(noiseDim),
    numFunctions(0),
    batchSize(batchSize),
    currentBatch(0),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    multiplier(multiplier),
    clippingParameter(clippingParameter),
    lambda(lambda),
    reset(false),
    deterministic(false),
    genWeights(0),
    discWeights(0)
{
  // Insert IdentityLayer for joining the Generator and Discriminator.
      this->discriminator.Model().insert(
      this->discriminator.Model().begin(),
      new IdentityLayer());
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::GAN(
    const GAN& network):
    predictors(network.predictors),
    responses(network.responses),
    generator(network.generator),
    discriminator(network.discriminator),
    initializeRule(network.initializeRule),
    noiseFunction(network.noiseFunction),
    noiseDim(network.noiseDim),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    multiplier(network.multiplier),
    clippingParameter(network.clippingParameter),
    lambda(network.lambda),
    reset(network.reset),
    currentBatch(network.currentBatch),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    noise(network.noise),
    deterministic(network.deterministic),
    genWeights(network.genWeights),
    discWeights(network.discWeights)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
 >
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::GAN(
    GAN&& network):
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    generator(std::move(network.generator)),
    discriminator(std::move(network.discriminator)),
    initializeRule(std::move(network.initializeRule)),
    noiseFunction(std::move(network.noiseFunction)),
    noiseDim(network.noiseDim),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    multiplier(network.multiplier),
    clippingParameter(network.clippingParameter),
    lambda(network.lambda),
    reset(network.reset),
    currentBatch(network.currentBatch),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    noise(std::move(network.noise)),
    deterministic(network.deterministic),
    genWeights(network.genWeights),
    discWeights(network.discWeights)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
 >
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::ResetData(
    InputType trainData)
{
  currentBatch = 0;

  numFunctions = trainData.n_cols;
  noise.set_size(noiseDim, batchSize);

  deterministic = false;
  ResetDeterministic();

  /**
   * These predictors are shared by the discriminator network. The additional
   * batch size predictors are taken from the generator network while training.
   * For more details please look in EvaluateWithGradient() function.
   */
  this->Predictors().set_size(trainData.n_rows, numFunctions + batchSize);
  this->Predictors().cols(0, numFunctions - 1) = std::move(trainData);
  this->discriminator.Predictors() = arma::mat(this->Predictors().memptr(),
      this->Predictors().n_rows, this->Predictors().n_cols, false, false);

  responses.ones(1, numFunctions + batchSize);
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);
  this->discriminator.Responses() = arma::mat(this->Responses().memptr(),
      this->Responses().n_rows, this->Responses().n_cols, false, false);

  this->generator.Predictors().set_size(noiseDim, batchSize);
  this->generator.Responses().set_size(predictors.n_rows, batchSize);

  if (!reset)
  {
    Reset();
  }
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
 >
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Reset()
{
  genWeights = 0;
  discWeights = 0;

  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);

  for (size_t i = 0; i < generator.network.size(); ++i)
  {
    genWeights += generator.Model()[i]->WeightSize();
  }

  for (size_t i = 0; i < discriminator.network.size(); ++i)
  {
    discWeights += discriminator.Model()[i]->WeightSize();
  }

  parameter.set_size(genWeights + discWeights, 1);
  generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
      false);
  discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
      discWeights, 1, false, false);

  // Initialize the parameters generator
  networkInit.Initialize(generator.Model(), parameter);
  // Initialize the parameters discriminator
  networkInit.Initialize(discriminator.Model(), parameter, genWeights);

  reset = true;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename OptimizerType, typename... CallbackTypes>
double GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Train(
    InputType trainData,
    OptimizerType& Optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(trainData));

  return Optimizer.Optimize(*this, parameter, callbacks...);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Evaluate(
    const OutputType& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  currentInput = arma::mat(predictors.memptr() + (i * predictors.n_rows),
      predictors.n_rows, batchSize, false, false);
  currentTarget = arma::mat(responses.memptr() + i, 1, batchSize, false,
      false);

  discriminator.Forward(currentInput);
  double res = discriminator.outputLayer.Forward(
      discriminator.Model().back()->OutputParameter(), currentTarget);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(noise);

  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      generator.network.back()->OutputParameter();
  discriminator.Forward(predictors.cols(numFunctions,
      numFunctions + batchSize - 1));
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);

  currentTarget = arma::mat(responses.memptr() + numFunctions,
      1, batchSize, false, false);
  res += discriminator.outputLayer.Forward(
      discriminator.Model().back()->OutputParameter(), currentTarget);

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename GradType, typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
EvaluateWithGradient(const OutputType& /* parameters */,
                     const size_t i,
                     GradType& gradient,
                     const size_t /* batchSize */)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (gradient.is_empty())
  {
    gradient = arma::zeros<OutputType>(parameter.n_elem, 1);
  }
  else
    gradient.zeros();

  if (this->deterministic)
  {
    this->deterministic = false;
    ResetDeterministic();
  }

  if (noiseGradientDiscriminator.is_empty())
  {
    noiseGradientDiscriminator = arma::zeros<OutputType>(
        gradientDiscriminator.n_elem, 1);
  }
  else
  {
    noiseGradientDiscriminator.zeros();
  }

  gradientGenerator = arma::mat(gradient.memptr(),
      generator.Parameters().n_elem, 1, false, false);

  gradientDiscriminator = arma::mat(gradient.memptr() +
      gradientGenerator.n_elem,
      discriminator.Parameters().n_elem, 1, false, false);

  // Get the gradients of the Discriminator.
  double res = discriminator.EvaluateWithGradient(discriminator.parameter,
      i, gradientDiscriminator, batchSize);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(noise);
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      generator.Model().back()->OutputParameter();
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);

  // Get the gradients of the Generator.
  res += discriminator.EvaluateWithGradient(discriminator.parameter,
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(noise))).
    // Pass the error from Discriminator to Generator.
    responses.cols(numFunctions, numFunctions + batchSize - 1) =
        arma::ones(1, batchSize);

    discriminator.outputLayer.Backward(
        discriminator.Model().back()->OutputParameter(),
        discriminator.Responses().cols(
        numFunctions, numFunctions + batchSize - 1),
        discriminator.error);
    discriminator.Backward();

    generator.error = discriminator.Model()[1]->Delta(); 

    generator.Predictors() = noise;
    generator.Backward();
    generator.ResetGradients(gradientGenerator);
    generator.Gradient(generator.Predictors().cols(0, batchSize - 1));

    gradientGenerator *= multiplier;
  }

  currentBatch++;


  if (preTrainSize > 0)
  {
    preTrainSize--;
  }

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
 >
template<typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, void>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
Gradient(const OutputType& parameters,
         const size_t i,
         OutputType& gradient,
         const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
 >
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Shuffle()
{
  const arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      numFunctions - 1, numFunctions));
  predictors.cols(0, numFunctions - 1) = predictors.cols(ordering);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Forward(
    const InputType& input)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  generator.Forward(input);
  arma::mat ganOutput = generator.Model().back()->OutputParameter();
  discriminator.Forward(ganOutput);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
Predict(InputType input, OutputType& output)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  Forward(input);

  output = discriminator.Model().back()->OutputParameter(); 
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
ResetDeterministic()
{
  
  // Reset Deterministic parameter for discriminator
  this->discriminator.Deterministic() = deterministic;
  this->discriminator.ResetDeterministic();

  //Reset Deterministic parameter for generator
  this->generator.Deterministic() = deterministic;
  this->generator.ResetDeterministic();
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename Archive>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(parameter));
  ar(CEREAL_NVP(generator));
  ar(CEREAL_NVP(discriminator));
  ar(CEREAL_NVP(reset));
  ar(CEREAL_NVP(genWeights));
  ar(CEREAL_NVP(discWeights));

  if (cereal::is_loading<Archive>())
  {
    // Share the parameters between the network.
    generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
        false);
    discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
        discWeights, 1, false, false);

    size_t offset = 0;
    for (size_t i = 0; i < generator.Model().size(); ++i)
    {
      generator.Model()[i]->Parameters() = arma::mat(generator.parameter.memptr() + offset,
          generator.Model()[i]->Parameters().n_rows,
          generator.Model()[i]->Parameters().n_cols, false, false);

      offset += generator.Model()[i]->Parameters().n_elem;

      //boost::apply_visitor(resetVisitor, generator.network[i]);
      generator.Model()[i]->Reset();
    }

    offset = 0;
    for (size_t i = 0; i < discriminator.Model().size(); ++i)
    {
      discriminator.Model()[i]->Parameters()= arma::mat(discriminator.parameter.memptr() + offset,
          discriminator.Model()[i]->Parameters().n_rows,
          discriminator.Model()[i]->Parameters().n_cols, false, false);

      offset += discriminator.Model()[i]->Parameters().n_elem;

      //boost::apply_visitor(resetVisitor, discriminator.network[i]);
      discriminator.Model()[i]->Reset();
    }

    deterministic = true;
    ResetDeterministic();
  }
}

} // namespace ann
} // namespace mlpack
# endif
