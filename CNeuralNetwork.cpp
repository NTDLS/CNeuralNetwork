///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CNeuralNetwork.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CNeuralNetwork::CNeuralNetwork(double learningRate, int hiddenLayers, int inputNeurons, int hiddenNeurons, int outputNeurons)
{
	this->LearningRate = learningRate;
	this->HiddenLayers = hiddenLayers;
	this->InputNeurons = inputNeurons;
	this->HiddenNeurons = hiddenNeurons;
	this->OutputNeurons = outputNeurons;
	this->TotalLayers = hiddenLayers + 2; //Count of hidden layers + input layer + output layer.

	memset(&this->Layers, 0, sizeof(this->Layers));

	srand(GetTickCount());

	this->Layers.Layer = (NNLAYER *) this->Memory.Allocate(this->TotalLayers, sizeof(NNLAYER));
	for(this->Layers.Count = 0; this->Layers.Count < this->TotalLayers; this->Layers.Count++)
	{
		NNLAYER *layer = &this->Layers.Layer[this->Layers.Count];

		int layerNeurons = 0;

		if(this->Layers.Count == 0) //The first layer is the input layer.
		{
			layerNeurons = this->InputNeurons;
		}
		else if(this->Layers.Count == this->TotalLayers - 1) //The last layer is the output layer.
		{
			layerNeurons = this->OutputNeurons;
		}
		else { //All other layers are hidden layers.
			layerNeurons = this->HiddenNeurons;
		}

		layer->Neurons.Neuron = (NNNEURON **) this->Memory.Allocate(layerNeurons, sizeof(NNNEURON *));

		for(layer->Neurons.Count = 0; layer->Neurons.Count < layerNeurons; layer->Neurons.Count++)
		{
			NNNEURON *neuron = (NNNEURON *) this->Memory.Allocate(1, sizeof(NNNEURON));
			layer->Neurons.Neuron[layer->Neurons.Count] = neuron;

			if(this->Layers.Count > 0) //The input neurons do not need a bias, their values are user supplied.
			{
				neuron->Bias = this->RandomValue(); //Bias is initially random.
			}
		}

		if(this->Layers.Count > 0)
		{
			//Connect each and every neuron in this layer to each and every neuron in the previous layer.
			NNLAYER *connectLayer = &this->Layers.Layer[this->Layers.Count - 1];
			for(int s = 0; s < connectLayer->Neurons.Count; s++)
			{
				connectLayer->Neurons.Neuron[s]->Dendrites.Count = layer->Neurons.Count;
				connectLayer->Neurons.Neuron[s]->Dendrites.Dendrite = (NNDENDRITE *) this->Memory.Allocate(layer->Neurons.Count, sizeof(NNDENDRITE));

				for(int t = 0; t < layer->Neurons.Count; t++)
				{
					connectLayer->Neurons.Neuron[s]->Dendrites.Dendrite[t].Weight = this->RandomValue(); //Weights are initially random.
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CNeuralNetwork::~CNeuralNetwork()
{
	for(int i = 0; i < this->Layers.Count; i++)
	{
		NNLAYER *layer = &this->Layers.Layer[i];

		for(int n = 0; n < layer->Neurons.Count; n++)
		{
			if(layer->Neurons.Neuron[n]->Dendrites.Dendrite)
			{
				this->Memory.Free(layer->Neurons.Neuron[n]->Dendrites.Dendrite);
			}
			this->Memory.Free(layer->Neurons.Neuron[n]);
		}
		this->Memory.Free(layer->Neurons.Neuron);
	}

	this->Memory.Free(this->Layers.Layer);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CNeuralNetwork::Train(double input[], double expectedOutput[])
{
	//Set the input values and populate all other neurons values.

	this->SetLayer(0, input);

	//double *dummyResult = new double[this->OutputNeurons];
	//Compute(input, dummyResult);
	//delete dummyResult;

	for(int l = this->TotalLayers - 1; l > 0; l--)
	{ 
		NNLAYER *layer = &this->Layers.Layer[l];

		for(int n = 0; n < layer->Neurons.Count; n++)
		{
			NNNEURON *neuron = layer->Neurons.Neuron[n];

			double Actual = neuron->Value;
			 
			if(l == this->TotalLayers - 1)
			{
				double Target = expectedOutput[n];
				neuron->Delta = (Actual * (1 - Actual) * (Target - Actual));
			}
			else
			{
				neuron->Delta = Actual * (1 - Actual) * this->SigmaWeightDelta(l, n);
			}

			for(int k = 0; k < this->Layers.Layer[l-1].Neurons.Count; k++)
			{
				NNNEURON *prevNeuron = this->Layers.Layer[l-1].Neurons.Neuron[k];
				prevNeuron->Dendrites.Dendrite[n].Weight += (neuron->Delta * this->LearningRate * prevNeuron->Value);
			}

			if(l != 0)
			{
				neuron->Bias += neuron->Delta * LearningRate;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CNeuralNetwork::SetLayer(int layer, double input[])
{
	for(int n = 0; n < this->Layers.Layer[layer].Neurons.Count; n++)
	{
		this->Layers.Layer[layer].Neurons.Neuron[n]->Value = input[n];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CNeuralNetwork::Compute(double input[], double *output)
{
	this->SetLayer(0, input);

    for(int i = 1; i < this->Layers.Count; i++)
    {
        for(int j = 0; j < this->Layers.Layer[i].Neurons.Count; j++)
        {
            this->Layers.Layer[i].Neurons.Neuron[j]->Value = 0;

            for(int k = 0; k < this->Layers.Layer[i-1].Neurons.Count; k++)
            {
                this->Layers.Layer[i].Neurons.Neuron[j]->Value =
					this->Layers.Layer[i].Neurons.Neuron[j]->Value
					+ this->Layers.Layer[i-1].Neurons.Neuron[k]->Value
					* this->Layers.Layer[i-1].Neurons.Neuron[k]->Dendrites.Dendrite[j].Weight;
            }

            this->Layers.Layer[i].Neurons.Neuron[j]->Value =
				this->Layers.Layer[i].Neurons.Neuron[j]->Value
				+ this->Layers.Layer[i].Neurons.Neuron[j]->Bias;

            this->Layers.Layer[i].Neurons.Neuron[j]->Value = this->Sigmoid(this->Layers.Layer[i].Neurons.Neuron[j]->Value);
        }
    }

	for(int n = 0; n < this->Layers.Layer[this->TotalLayers - 1].Neurons.Count; n++)
	{
		output[n] = this->Layers.Layer[this->TotalLayers - 1].Neurons.Neuron[n]->Value;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double CNeuralNetwork::Sigmoid(double value)
{ 
    return (1.0/(1.0+exp(-value)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Return a random number between range -1.0 to 1.0
double CNeuralNetwork::RandomValue(void)
{
	double r = ((((rand() % 100) + 1) / 101.0) * (-(rand() % 100 >= 50 ? 1 : -1))) * 0.5;
	return r;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double CNeuralNetwork::SigmaWeightDelta(unsigned long layerIndex, unsigned long neuronIndex)
{
    double result = 0.0;

	NNLAYER *layer = &this->Layers.Layer[layerIndex];
	NNNEURON *neuron = layer->Neurons.Neuron[neuronIndex];

	for(int i = 0; i < this->Layers.Layer[layerIndex + 1].Neurons.Count; i++)
    {
		NNLAYER *nextLayer = &this->Layers.Layer[layerIndex + 1];
		NNNEURON *nextNeuron = nextLayer->Neurons.Neuron[i];

        result += neuron->Dendrites.Dendrite[i].Weight * nextNeuron->Delta;
    }

	return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
