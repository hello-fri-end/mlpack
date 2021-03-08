/**
 * @file gan.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_GAN_HPP
#define MLPACK_METHODS_ANN_GAN_GAN_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/gan/gan_policies.hpp>


namespace mlpack {
namespace ann /** Artificial Neural Network. **/ {

/**
 * The implementation of the standard GAN module. Generative Adversarial
 * Networks (GANs) are a class of artificial intelligence algorithms used
 * in unsupervised machine learning, implemented by a system of two neural
 * networks contesting with each other in a zero-sum game framework. This
 * technique can generate photographs that look at least superficially
 * authentic to human observers, having many realistic characteristics.
 * GANs have been used in Text-to-Image Synthesis, Medical Drug Discovery,
 * High Resolution Imagery Generation, Neural Machine Translation and so on.
 *
 * For more information, see the following paper:
 *
 * @code
 * @article{Goodfellow14,
 *   author    = {Ian J. Goodfellow, Jean Pouget-Abadi, Mehdi Mirza, Bing Xu,
 *                David Warde-Farley, Sherjil Ozair, Aaron Courville and
 *                Yoshua Bengio},
 *   title     = {Generative Adversarial Nets},
 *   year      = {2014},
 *   url       = {http://arxiv.org/abs/1406.2661},
 *   eprint    = {1406.2661},
 * }
 * @endcode
 *
 * @tparam Model The class type of Generator and Discriminator.
 * @tparam InitializationRuleType Type of Initializer.
 * @tparam Noise The noise function to use.
 * @tparam PolicyType The GAN variant to be used (GAN, DCGAN, WGAN or WGANGP).
 */
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType = StandardGAN,
  typename InputType = arma::mat,
  typename OutputType = arma::mat,
>
class GAN
{
 public:
  /**
   * Constructor for GAN class.
   *
   * @param generator Generator network.
   * @param discriminator Discriminator network.
   * @param batchSize Batch size to be used for training.
   * @param generatorUpdateStep Number of steps to train Discriminator
   *                            before updating Generator.
   * @param preTrainSize Number of pre-training steps of Discriminator.
   * @param multiplier Ratio of learning rate of Discriminator to the Generator.
   * @param clippingParameter Weight range for enforcing Lipschitz constraint.
   * @param lambda Parameter for setting the gradient penalty.
   */
  GAN(Model generator,
      Model discriminator,
      InitializationRuleType& initializeRule,
      Noise& noiseFunction,
      const size_t noiseDim,
      const size_t batchSize,
      const size_t generatorUpdateStep,
      const size_t preTrainSize,
      const double multiplier,
      const double clippingParameter = 0.01,
      const double lambda = 10.0);

  //! Copy constructor.
  GAN(const GAN&);

  //! Move constructor.
  GAN(GAN&&);

  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param predictors The data points of real distribution.
   */
  void ResetData(InputType predictors);

  // Reset function.
  void Reset();

  /**
   * Train function.
   *
   * @param predictors The data points of real distribution.
   * @param optimizer Instantiated optimizer used to train the model.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename Policy = PolicyType, typename OptimizerType>
  typename std::enable_if<std::is_same<Policy, WGANGP>::value, double>::type
  Train(InputType predictors, OptimizerType& optimizer);

  /**
   * Train function for the Standard GAN and DCGAN.
   *
   * @param predictors The data points of real distribution.
   * @param discriminatorOptimizer Optimizer for discriminator network.
   * @param generatorOptimizer Optimizer for generator network.
   * @param maxIterations Number of iterations for which to train GAN.
   */
  template<typename Policy = PolicyType,
           typename DiscOptimizerType,
           typename GenOptimizerType>
  typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                          std::is_same<Policy, DCGAN>::value, void>::type
  Train(InputType predictors,
        DiscOptimizerType& discriminatorOptimizer,
        GenOptimizerType& generatorOptimizer,
        size_t maxIterations);

  /**
   * Train function for WGAN.
   *
   * @param predictors The data points of real distribution.
   * @param discriminatorOptimizer Optimizer for discriminator network.
   * @param generatorOptimizer Optimizer for generator network.
   * @param maxIterations Number of iterations for which to train GAN.
   * @param discIterations Number of discriminator iterations in one iteration
   *        of WGAN (Default: one iteration).
   */
  template<typename Policy = PolicyType,
           typename DiscOptimizerType,
           typename GenOptimizerType>
  typename std::enable_if<std::is_same<Policy, WGAN>::value, void>::type
  Train(InputType predictors,
        DiscOptimizerType& discriminatorOptimizer,
        GenOptimizerType& generatorOptimizer,
        size_t maxIterations,
        size_t discIterations = 1);

  /**
   * Evaluate function for the WGAN-GP.
   * This function gives the performance of the WGAN-GP on the current input.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                          double>::type
  Evaluate(const InputType& parameters,
           const size_t i,
           const size_t batchSize);

  /**
   * EvaluateWithGradient function for the WGAN-GP.
   * This function gives the performance of the WGAN-GP on the
   * current input, while updating Gradients.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename GradType, typename Policy = PolicyType>
  typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                          double>::type
  EvaluateWithGradient(const InputType& parameters,
                       const size_t i,
                       GradType& gradient,
                       const size_t batchSize);

  /**
   * Gradient function for WGAN-GP.
   * This function passes the gradient based on which network is being
   * trained, i.e., Generator or Discriminator.
   *
   * @param parameters present parameters of the network.
   * @param i Index of the predictors.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                          void>::type
  Gradient(const InputType& parameters,
           const size_t i,
           OutputType& gradient,
           const size_t batchSize);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  /**
   * This function does a forward pass through the GAN network.
   *
   * @param input Sampled noise.
   */
  void Forward(InputType&& input);

  /**
   * This function predicts the output of the network on the given input.
   *
   * @param input The input the Discriminator network.
   * @param output Result of the Discriminator network.
   */
  void Predict(InputType&& input,
              OutputType& output);

  //! Return the parameters of the network.
  const OutputType& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  OutputType& Parameters() { return parameter; }

  //! Return the generator of the GAN.
  const Model& Generator() const { return generator; }
  //! Modify the generator of the GAN.
  Model& Generator() { return generator; }
  //! Return the discriminator of the GAN.
  const Model& Discriminator() const { return discriminator; }
  //! Modify the discriminator of the GAN.
  Model& Discriminator() { return discriminator; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Get the matrix of responses to the input data points.
  const arma::mat& Responses() const { return responses; }
  //! Modify the matrix of responses to the input data points.
  arma::mat& Responses() { return responses; }

  //! Get the matrix of data points (predictors).
  const InputType& Predictors() const { return predictors; }
  //! Modify the matrix of data points (predictors).
  InputType& Predictors() { return predictors; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
  * Reset the module status by setting the current deterministic parameter
  * for the discriminator and generator networks and their respective layers.
  */
  void ResetDeterministic();
  //! Locally stored parameter for training data + noise data.
  InputType predictors;
  //! Locally stored parameters of the network.
  OutputType parameter;
  //! Locally stored Generator network.
  Model generator;
  //! Locally stored Discriminator network.
  Model discriminator;
  //! Locally stored Initializer.
  InitializationRuleType initializeRule;
  //! Locally stored Noise function
  Noise noiseFunction;
  //! Locally stored input dimension of the Generator network.
  size_t noiseDim;
  //! Locally stored number of data points.
  size_t numFunctions;
  //! Locally stored batch size parameter.
  size_t batchSize;
  //! Locally stored number of iterations that have been completed.
  size_t counter;
  //! Locally stored batch number which is being processed.
  size_t currentBatch;
  //! Locally stored number of training step before Generator is trained.
  size_t generatorUpdateStep;
  //! Locally stored number of pre-train step for Discriminator.
  size_t preTrainSize;
  //! Locally stored learning rate ratio for Generator network.
  double multiplier;
  //! Locally stored weight clipping parameter.
  double clippingParameter;
  //! Locally stored lambda parameter.
  double lambda;
  //! Locally stored reset parameter.
  bool reset;
  //! Locally stored responses.
  OutputType responses;
  //! Locally stored current input.
  InputType currentInput;
  //! Locally stored current target.
  OutputType currentTarget;
  //! Locally stored gradient parameters.
  OutputType gradient;
  //! Locally stored gradient for Discriminator.
  OutputType gradientDiscriminator;
  //! Locally stored gradient for noise data in the predictors.
  OutputType noiseGradientDiscriminator;
  //! Locally stored norm of the gradient of Discriminator.
  OutputType normGradientDiscriminator;
  //! Locally stored noise using the noise function.
  InputType noise;
  //! Locally stored gradient for Generator.
  OutputType gradientGenerator;
  //! The current evaluation mode (training or testing).
  bool deterministic;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gan_impl.hpp"
#include "wgan_impl.hpp"
#include "wgangp_impl.hpp"


#endif
