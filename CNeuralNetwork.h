///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../@Libraries/CMemPool/CMemPool.H"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
	//Dendrites are referenced by index and do not require pointers to child nuerons.
	double Weight; //Weight of the conection to the neuron.
} NNDENDRITE;

typedef struct {
	NNDENDRITE *Dendrite;
	int Count;
} NNDENDRITES;

typedef struct {
	NNDENDRITES Dendrites;
    double Value; //Value which Neuron currently is holding
    double Bias;  //Bias of the neuron
    double Delta; //Used in back-propagation. Note it is back-propagation specific.
} NNNEURON;

typedef struct {
	NNNEURON **Neuron;
	int Count;
} NNNEURONS;

typedef struct {
	NNNEURONS Neurons;
} NNLAYER;

typedef struct {
	NNLAYER *Layer;
	int Count;
} NNLAYERS;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CNeuralNetwork {
public:
	CNeuralNetwork(double learningRate, int hiddenLayers, int inputNeurons, int hiddenNeurons, int outputNeurons);
	~CNeuralNetwork();

	void Train(double input[], double expectedOutput[]);
	void Compute(double input[], double *output);
	void SetLayer(int layer, double input[]);

	double GetLearningRate(void)
	{
		return this->LearningRate;
	}

	void SetLearningRate(double learningRate)
	{
		this->LearningRate = learningRate;
	}

private:
	NNLAYERS Layers;
	CMemPool Memory;

	double LearningRate; //Learning rate of the network.
	int HiddenLayers; //The number of hidden layers.
	int InputNeurons; //The number of neurons in the input layer.
	int HiddenNeurons; //The number of neurons in the hidden layer(s).
	int OutputNeurons; //The number of neurons in the output layer.
	int TotalLayers; //The total number of layers in the collection.

	double Sigmoid(double value);
	double RandomValue(void);
	double SigmaWeightDelta(unsigned long layerIndex, unsigned long neuronIndex);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
