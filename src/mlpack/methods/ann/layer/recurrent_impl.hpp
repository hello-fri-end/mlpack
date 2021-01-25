/**
 * @file methods/ann/layer/recurrent_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Recurrent class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent.hpp"


namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
RecurrentType<InputType, OutputType>::RecurrentType() :
    rho(0),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false),
    ownsLayer(false)
{
  // Nothing to do.
}

template <typename InputType, typename OutputType>
template<
    typename StartModuleType,
    typename InputModuleType,
    typename FeedbackModuleType,
    typename TransferModuleType
>
RecurrentType<InputType, OutputType>::RecurrentType(
    const StartModuleType& start,
    const InputModuleType& input,
    const FeedbackModuleType& feedback,
    const TransferModuleType& transfer,
    const size_t rho) :
    startModule(new StartModuleType(start)),
    inputModule(new InputModuleType(input)),
    feedbackModule(new FeedbackModuleType(feedback)),
    transferModule(new TransferModuleType(transfer)),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false),
    ownsLayer(true)
{
  initialModule = new Sequential();
  mergeModule = new AddMerge(false, false, false);
  recurrentModule = new Sequential(false, false);

  //boost::apply_visitor(AddVisitor<CustomLayers...>(inputModule),initialModule);
  initialModule->Add(inputModule);

  //boost::apply_visitor(AddVisitor<CustomLayers...>(startModule),
    //                   initialModule);

  initialModule->Add(startModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(transferModule),
     //                  initialModule);
  initialModule->Add(transferModule);

  //boost::apply_visitor(AddVisitor<CustomLayers...>(inputModule), mergeModule);
  mergeModule->Add(inputModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(feedbackModule),
    //                   mergeModule);
  mergeModule->Add(feedbackModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(mergeModule),
                       //recurrentModule);
  recurrentModule->Add(mergeModule);

  //boost::apply_visitor(AddVisitor<CustomLayers...>(transferModule),
  //                     recurrentModule);
  recurrentModule->Add(transferModule);

  network.push_back(initialModule);
  network.push_back(mergeModule);
  network.push_back(feedbackModule);
  network.push_back(recurrentModule);
}

template<typename InputType, typename OutputType>
RecurrentType<InputType, OutputType>::RecurrentType(
    const RecurrentType& network) :
    rho(network.rho),
    forwardStep(network.forwardStep),
    backwardStep(network.backwardStep),
    gradientStep(network.gradientStep),
    deterministic(network.deterministic),
    ownsLayer(network.ownsLayer),
    startModule(network.startModule),
    inputModule(network.inputModule),
    feedbackModule(feedbackModule),
    transferModule(transferModule)
{
  //startModule = boost::apply_visitor(copyVisitor, network.startModule);
  //startModule = new network->startModule(network.startModule);

  //inputModule = boost::apply_visitor(copyVisitor, network.inputModule);
  //inputModule = new InputModuleType(network.inputModule);
  //feedbackModule = boost::apply_visitor(copyVisitor, network.feedbackModule);
  //feedbackModule = new FeedbackModuleType(network.feedbackModule);
  //transferModule = boost::apply_visitor(copyVisitor, network.transferModule);
  //transferModule = new TransferModuleType(network.transferModule);
  //initialModule = new Sequential<>();
  //mergeModule = new AddMerge<>(false, false, false);
  //recurrentModule = new Sequential<>(false, false);

  //boost::apply_visitor(AddVisitor<CustomLayers...>(inputModule),
    //                   initialModule);
  initialModule->Add(inputModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(startModule),
   //                    initialModule);
   initialModule->Add(startModule);                    
  //boost::apply_visitor(AddVisitor<CustomLayers...>(transferModule),
  //                   initialModule);
  initialModule->Add(transferModule);

  //boost::apply_visitor(AddVisitor<CustomLayers...>(inputModule), mergeModule);
  mergeModule->Add(inputModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(feedbackModule),
  //                     mergeModule);
  mergeModule->Add(feedbackModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(mergeModule),
  //                     recurrentModule);
  recurrentModule->Add(mergeModule);
  //boost::apply_visitor(AddVisitor<CustomLayers...>(transferModule),
   //                    recurrentModule);
  recurrentModule->Add(transferModule);
  this->network.push_back(initialModule);
  this->network.push_back(mergeModule);
  this->network.push_back(feedbackModule);
  this->network.push_back(recurrentModule);
}

template<typename InputType, typename OutputType>
void RecurrentType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (forwardStep == 0)
  {
    //boost::apply_visitor(ForwardVisitor(input, output), initialModule);
    initialModule->Forward(input,output);
  }
  else
  {
    //boost::apply_visitor(ForwardVisitor(input,
        //boost::apply_visitor(outputParameterVisitor, inputModule)),
        //inputModule);
    inputModule->Forward(input, inputModule->OutputParameter()); 

    //boost::apply_visitor(ForwardVisitor(boost::apply_visitor(
        //outputParameterVisitor, transferModule),
        //boost::apply_visitor(outputParameterVisitor, feedbackModule)),
        //feedbackModule);
    feedbackModule->Forward(transferModule->OutputParameter(), feedbackModule->OutputParameter()); 

    //boost::apply_visitor(ForwardVisitor(input, output), recurrentModule);
    recurrentModule->Forward(input,output);
  }

  //output = boost::apply_visitor(outputParameterVisitor, transferModule);
  output= transferModule->OutputParameter();

  // Save the feedback output parameter when training the module.
  if (!deterministic)
  {
    feedbackOutputParameter.push_back(output);
  }

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    backwardStep = 0;

    if (!recurrentError.is_empty())
    {
      recurrentError.zeros();
    }
  }
}

template<typename InputType, typename OutputType>
void RecurrentType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  if (!recurrentError.is_empty())
  {
    recurrentError += gy;
  }
  else
  {
    recurrentError = gy;
  }

  if (backwardStep < (rho - 1))
  {
    //boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        //outputParameterVisitor, recurrentModule), recurrentError,
        //boost::apply_visitor(deltaVisitor, recurrentModule)),
        //recurrentModule);
      recurrentModule->Backward(recurrentModule->OutputParameter(), recurrentError,recurrentModule->Delta());

    //boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        //outputParameterVisitor, inputModule),
        //boost::apply_visitor(deltaVisitor, recurrentModule), g),
        //inputModule);
      inputModule->Backward(inputModule->OutputParameter(), recurrentModule->Delta(), g);

    //boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        //outputParameterVisitor, feedbackModule),
        //boost::apply_visitor(deltaVisitor, recurrentModule),
        //boost::apply_visitor(deltaVisitor, feedbackModule)), feedbackModule);
      feedbackModule->Backward(feedbackModule->OutputParameter(), recurrentModule->Delta(), feedbackModule->Delta());
  }
  else
  {
    //boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        //outputParameterVisitor, initialModule), recurrentError, g),
        //initialModule);
    initialModule->Backward(initialModule->OutputParameter(), recurrentError, g);
  }

  //recurrentError = boost::apply_visitor(deltaVisitor, feedbackModule);
  recurrentError = feedbackModule->Delta();

  backwardStep++;
}

template<typename InputType, typename OutputType>
void RecurrentType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */)
{
  if (gradientStep < (rho - 1))
  {
    //boost::apply_visitor(GradientVisitor(input, error), recurrentModule);
    recurrentModule->Gradient(input, error);

    //boost::apply_visitor(GradientVisitor(input,
        //boost::apply_visitor(deltaVisitor, mergeModule)), inputModule);
    inputModule->Gradient(input, mergeModule->Delta());


    //boost::apply_visitor(GradientVisitor(
       // feedbackOutputParameter[feedbackOutputParameter.size() - 2 -
       // gradientStep], boost::apply_visitor(deltaVisitor,
      //  mergeModule)), feedbackModule);
      feedbackModule->Gradient(feedbackOutputParameter[feedbackOutputParameter.size() - 2 - gradientStep], mergeModule->Delta());
  }
  else
  {
    //boost::apply_visitor(GradientZeroVisitor(), recurrentModule);
    recurrentModule->Gradient().zeros();
    //boost::apply_visitor(GradientZeroVisitor(), inputModule);
    inputModule->Gradient().zeros();
    //boost::apply_visitor(GradientZeroVisitor(), feedbackModule);
    feedbackModule->Gradient().zeros();

    //boost::apply_visitor(GradientVisitor(input,
        //boost::apply_visitor(deltaVisitor, startModule)), initialModule);
      initialModule->Gradient(input, startModule->Delta());
  }

  gradientStep++;
  if (gradientStep == rho)
  {
    gradientStep = 0;
    feedbackOutputParameter.clear();
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void RecurrentType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // Clean up memory, if we are loading.
  if (cereal::is_loading<Archive>())
  {
    // Clear old things, if needed.
    //boost::apply_visitor(DeleteVisitor(), recurrentModule);
    delete *recurrentModule;
    //boost::apply_visitor(DeleteVisitor(), initialModule);
    delete *initialModule;
    //boost::apply_visitor(DeleteVisitor(), startModule);
    delete *startModule;
    network.clear();
  }

  ar(CEREAL_VARIANT_POINTER(startModule));
  ar(CEREAL_VARIANT_POINTER(inputModule));
  ar(CEREAL_VARIANT_POINTER(feedbackModule));
  ar(CEREAL_VARIANT_POINTER(transferModule));
  ar(CEREAL_NVP(rho));
  ar(CEREAL_NVP(ownsLayer));

  // Set up the network.
  if (cereal::is_loading<Archive>())
  {
    initialModule = new Sequential();
    mergeModule = new AddMerge(false, false, false);
    recurrentModule = new Sequential(false, false);

    //boost::apply_visitor(AddVisitor<CustomLayers...>(inputModule),
    //                     initialModule);
    initialModule->Add(inputModule);
    //boost::apply_visitor(AddVisitor<CustomLayers...>(startModule),
     //                    initialModule);
    initialModule->Add(startModule);
    //boost::apply_visitor(AddVisitor<CustomLayers...>(transferModule),
      //                   initialModule);
    initialModule->Add(transferModule);

    //boost::apply_visitor(AddVisitor<CustomLayers...>(inputModule),
    //                     mergeModule);
    mergeModule->Add(inputModule);
    //boost::apply_visitor(AddVisitor<CustomLayers...>(feedbackModule),
    //                     mergeModule);
    mergeModule->Add(feedbackModule);
    //boost::apply_visitor(AddVisitor<CustomLayers...>(mergeModule),
     //                    recurrentModule);
     recurrentModule->Add(mergeModule);
    //boost::apply_visitor(AddVisitor<CustomLayers...>(transferModule),
     //                    recurrentModule);
     recurrentModule->Add(transferModule);

    network.push_back(initialModule);
    network.push_back(mergeModule);
    network.push_back(feedbackModule);
    network.push_back(recurrentModule);
  }
}

} // namespace ann
} // namespace mlpack

#endif
