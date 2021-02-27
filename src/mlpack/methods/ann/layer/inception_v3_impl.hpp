/**
 * @file inception_v3_impl.hpp
 * @author Shah Anwaar Khalid
 *
 * Implementation of the InceptionV3 Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it uder the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_V3_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_V3_IMPL_HPP

// In case it is not included.
#include "inception_v3.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputType, typename OutputType, int module>
Inception3<InputType, OutputType, module> :: Inception3(
    const size_t inSize,
    const size_t inputWidth,
    const size_t inputHeight,
    const arma::vec outA,
    const arma::vec outB,
    const arma::vec outC,
    const arma::vec outD,
    const arma::vec outE,
    const arma::vec outF)
{
  model = new Concat(true);

  //! Build the Inception3A module 
  if(module == 1)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<Convolution>(inSize, outA[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkA->Add<BatchNorm>(outA[0]);
    networkA->Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    networkB->Add<Convolution>(outB[0], outB[1], 3, 3, 1, 1, 0, 0,
       inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[1]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 3, 3, 1, 1, 0, 0,
        inputWidth - 2, inputHeight - 2);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

    //! Build Network D
    Sequential* networkD;

    networkD->Add<MaxPooling>(3, 3, 1, 1);
    
    networkD->Add<Convolution>(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth - 2 , inputHeight - 2);
    networkD->Add<BatchNorm>(outD[0]);
    networkD->Add<ReLULayer>();

    model->Add(networkD);
  }
  //! Build Inception Block B module
  else if(module == 2)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<Convolution>(inSize, outA[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkA->Add<BatchNorm>(outA[0]);
    networkA->Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<AveragePooling>(3, 3, 1, 1);

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1,
        inputWidth - 2, inputHeight -2);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 1, 7, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 7, 1, 1, 1, 0, 0,
        inputWidth, inputHeight - 6);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

    //! Build Network D
    Sequential* networkD;
    networkD = new Sequential();

    networkD->Add<Convolution>(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[0]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[0], outD[1], 7, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[1]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[1], outD[2], 1, 7, 1, 1, 0, 0,
        inputWidth - 6, inputHeight);
    networkD->Add<BatchNorm>(outD[2]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[2], outD[3], 7, 1, 1, 1, 0, 0,
        inputWidth - 6, inputHeight - 6);
    networkD->Add<BatchNorm>(outD[3]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[3], outD[4], 1, 7, 1, 1, 0, 0,
        inputWidth - 12, inputHeight - 6);
    networkD->Add<BatchNorm>(outD[4]);
    networkD->Add<ReLULayer>();

    model->Add(networkD);
  }
  //! Build Inception Block C module
  else if(module == 3)
  {
    //! Build Network A
    Sequential* networkA; 
    networkA = new Sequential();

    networkA->Add<Convolution>(inSize, outA[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkA->Add<BatchNorm>(outA[0]);
    networkA->Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    networkB->Add<Convolution>(outB[0], outB[1], 1, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[1]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 3, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

    //! Build Network D
    Sequential* networkD;
    networkD = new Sequential();

    networkD->Add<Convolution>(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[0]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[0], outD[1], 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[1]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[1], outD[2], 3, 1, 1, 1, 0, 0,
        inputWidth - 2, inputHeight - 2);
    networkD->Add<BatchNorm>(outD[2]);
    networkD->Add<ReLULayer>();

    model->Add(networkD);

    //! Build Network E
    Sequential* networkE;
    networkE = new Sequential();

    networkE->Add<Convolution>(inSize, outE[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkE->Add<BatchNorm>(outE[0]);
    networkE->Add<ReLULayer>();

    networkE->Add<Convolution>(outE[0], outE[1], 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkE->Add<BatchNorm>(outE[1]);
    networkE->Add<ReLULayer>();

    networkE->Add<Convolution>(outE[1], outE[2], 1, 3, 1, 1, 0, 0,
        inputWidth - 2, inputHeight - 2);
    networkE->Add<BatchNorm>(outE[2]);
    networkE->Add<ReLULayer>();

    model->Add(networkE);

    //! Build Network F
    Sequential* networkF;
    networkF = new Sequential();

    networkF->Add<Maxpooling>(3, 3, 1, 1);

    networkF->Add<Convolution>(inSize, outF[0], 1, 1, 1, 1, 0, 0,
        inputWidth - 2, inputHeight - 2);
    networkF->Add<BatchNorm>(outF[0]);
    networkF->Add<ReLULayer>();

    model->Add(networkF);
  }
  //! Build Reduction Block A module
  else if(module == 4)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<Maxpooling>(3, 3, 1, 1);

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 3, 3, 1, 1, 0, 0,
        inputWidth - 2, inputHeight -2);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

  }
  //! Build Reduction Block B module
  else if(module ==5)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<Maxpooling>(3, 3, 1, 1);

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    networkB->Add<Convolution>(outB[0], outB[1], 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);                                    
    networkB->Add<BatchNorm>(outB[1]);
    networkB->Add<ReLULayer>();                                

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 1, 7, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 7, 1, 1, 1, 0, 0,
        inputWidth, inputHeight - 6);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[2], outC[3], 3, 3, 1, 1, 0, 0,
        inputWidth - 6, inputHeight - 6);
    networkC->Add<BatchNorm>(outC[3]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

  }
}

template<typename InputType, typename OutputType, int module>
Inception3<InputType, OutputType, module>::~Inception3()
{
  delete model;
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Forward(const InputType& input,
                                                    OutputType& output)
{
  model->Forward(std::move(input), std::move(output));
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Backward(const InputType& input,
                                                          const OutputType& gy,
                                                          OutputType &g)
{
  model->Backward(std::move(input), std::move(gy), std::move(g));
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  model->Gradient(std::move(input), std::move(error), std::move(gradient));
}

template<typename InputType, typename OutputType, int module>
template<typename Archive>
void Inception3<InputType, OutputType, module>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // Don't know how this works..yet
}

} // namespace ann
} // namespace mlpack

#endif
}
}
