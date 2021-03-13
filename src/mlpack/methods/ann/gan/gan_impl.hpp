/**
 * @file gan_impl.hpp
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
    batchSize(batchSize),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    multiplier(multiplier),
    clippingParameter(clippingParameter),
    lambda(lambda),
    reset(false),
    deterministic(false)
{
  // Insert IdentityLayer for joining the Generator and Discriminator.
  this->discriminator.network.insert(
      this->discriminator.network.begin(),
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
    counter(network.counter),
    currentBatch(network.currentBatch),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    noise(network.noise),
    deterministic(network.deterministic)
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
    counter(network.counter),
    currentBatch(network.currentBatch),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    noise(std::move(network.noise)),
    deterministic(network.deterministic)
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
  InputType predictors)
{
  currentBatch = 0;

  numFunctions = predictors.n_cols;
  noise.set_size(noiseDim, batchSize);

  deterministic = false;
  ResetDeterministic();

  this->predictors.set_size(predictors.n_rows, numFunctions + batchSize);
  this->predictors.cols(0, numFunctions - 1) = std::move(predictors);
  this->discriminator.predictors = arma::mat(this->predictors.memptr(),
      this->predictors.n_rows, this->predictors.n_cols, false, false);

  responses.ones(1, numFunctions + batchSize);
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);
  this->discriminator.responses = arma::mat(this->responses.memptr(),
      this->responses.n_rows, this->responses.n_cols, false, false);

  this->generator.predictors.set_size(noiseDim, batchSize);
  this->generator.responses.set_size(predictors.n_rows, batchSize);

  if ((!reset))
    Reset();
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
    genWeights += generator.network[i]->WeightSize();
  }

  for (size_t i = 0; i < discriminator.network.size(); ++i)
  {
    discWeights += discriminator.network[i]->WeightSize();
  }

  parameter.set_size(genWeights + discWeights, 1);
  generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
      false);
  discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
      discWeights, 1, false, false);

  // Initialize the parameters generator
  networkInit.Initialize(generator.network, parameter);
  // Initialize the parameters discriminator
  networkInit.Initialize(discriminator.network, parameter, genWeights);

  reset = true;
  generator.reset = true;
  discriminator.reset = true;
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
    InputType predictors,
    OptimizerType& Optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors));

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
template<typename Policy,
         typename DiscOptimizerType,
         typename GenOptimizerType,
         typename... CallbackTypes>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Train(
    InputType predictors,
    DiscOptimizerType& discriminatorOptimizer,
    GenOptimizerType& generatorOptimizer,
    size_t maxIterations,
    CallbackTypes&&... callbacks)
{

  ResetData(std::move(predictors));
  // To keep track of where we are.
  size_t currentFunction = 0;
  double objValue = 0;

  // We pass two batches during training hence maxIterations is doubled.
  discriminatorOptimizer.MaxIterations() =
      discriminatorOptimizer.MaxIterations() * 2;

  // TODO: find a way to allow shuffling
  // Avoid shuffling of predictors during training.
  discriminatorOptimizer.Shuffle() = false;
  generatorOptimizer.Shuffle() = false;

  // Predictors and responses for generator and discriminator network.
  InputType discriminatorPredictors;
  OutputType discriminatorResponses;

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; i++)
  {
    // Is this iteration the start of a sequence?
    if (currentFunction % numFunctions == 0 && i > 0)
    {
      currentFunction = 0;
    }

    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of functions left.
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions -
        currentFunction);

    // Training data for dicriminator.
    if (effectiveBatchSize != batchSize)
    {
      noise.set_size(noiseDim, effectiveBatchSize);
      discriminatorOptimizer.BatchSize() = effectiveBatchSize;
      // I think this should just be 2, we'll see
      discriminatorOptimizer.MaxIterations() = effectiveBatchSize * 2;
    }
    noise.imbue( [&]() { return noiseFunction();} );
    OutputType fakeImages;
    generator.Forward(noise, fakeImages);

    discriminatorPredictors = arma::join_rows(
        this->predictors.cols(currentFunction,  currentFunction + effectiveBatchSize -
        1), fakeImages);

    discriminatorResponses = arma::join_rows(arma::ones(1, effectiveBatchSize),
        arma::zeros(1, effectiveBatchSize));

   // Train the discriminator.
    objValue += discriminator.Train(discriminatorPredictors, discriminatorResponses,
        discriminatorOptimizer, callbacks...);
std::cout<<"predifctors check"<<std::endl;
  std::cout<<discriminator.predictors.n_rows<<" "<<discriminator.predictors.n_cols<<std::endl;

    if (effectiveBatchSize != batchSize)
    {
      noise.set_size(noiseDim, batchSize);
      discriminatorOptimizer.BatchSize() = batchSize;
      // same here?
      discriminatorOptimizer.MaxIterations() = batchSize * 2;
    }

    if (preTrainSize == 0)
    {
      // Calculate error for generator network.
      // Do we really need another forward pass 
      // through the generator? Can't we use the above fwd pass?
      discriminatorResponses = arma::ones(1, batchSize);

      noise.imbue( [&]() { return noiseFunction();} );
      generator.Forward(std::move(noise));

      discriminator.Forward(std::move
            (generator.network.back()->OutputParameter()));

      discriminator.outputLayer.Backward(
          std::move(discriminator.network.back()->OutputParameter()),
          std::move(discriminatorResponses),
          discriminator.error);
      discriminator.Backward();

      generator.error = discriminator.network[1]->Delta();

      // Train the generator network.
      objValue += generator.Train(noise, generatorOptimizer, callbacks...);
    std::cout<<"dual optimizer train function -- gan"<<std::endl;
    }

    if (preTrainSize > 0)
    {
      preTrainSize--;
    }

    currentFunction += effectiveBatchSize;
  }

  // Changing maxIterations back to normal.
  discriminatorOptimizer.MaxIterations() =
      discriminatorOptimizer.MaxIterations() / 2;
  return objValue;
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
  if (!reset)
    Reset();

  generator.Forward(std::move(input));
  arma::mat ganOutput = generator.network.back()->OutputParameter();

  discriminator.Forward(std::move(ganOutput));
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
Predict(InputType& input, OutputType& output)
{
  if (!reset)
    Reset();

  Forward(std::move(input));

  output = discriminator.network.back()->OutputParameter();
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
  this->discriminator.deterministic = deterministic;
  this->generator.deterministic = deterministic;
  this->discriminator.ResetDeterministic();
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
serialize(Archive& ar, const unsigned int /* version */)
 
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
