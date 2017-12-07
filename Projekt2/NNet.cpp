#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct Connection
{
	double weight;
	double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex, double weight[]);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer, unsigned layerNum);

private:
	static double tansigFunction(double x);
	static double purelinFunction(double x);
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

double Neuron::tansigFunction(double x)
{
	double tansig = (2 / (1 + exp(-2 * x))) -1;
	//cout << x << endl;
	//cout << "tansig(" << x << ") = " << tansig << endl;
	return tansig;
}

double Neuron::purelinFunction(double x)
{
	double lin = x;
	//cout << "purelin(" << x << ") = " << lin << endl;
	return lin;
}

void Neuron::feedForward(const Layer &prevLayer, unsigned layerNum)
{
	double sum = 0.0;
	unsigned n = 0;

	switch (layerNum)
	{
	case 1:
		for (n = 0; n < prevLayer.size(); ++n) {
			sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
			//cout << layerNum << " sum = " << sum << endl;
		}
		m_outputVal = Neuron::tansigFunction(sum);
		cout << "tansig = " << m_outputVal << endl;
		break;
	case 2:
		for (n = 0; n < prevLayer.size(); ++n) {
			sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
			//cout << layerNum << " sum = " << sum << endl;
		}
		m_outputVal = Neuron::purelinFunction(sum);
		cout << "purelin = " << m_outputVal << endl;
	default:
		break;
	}
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex, double weight[])
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = weight[c];
		//cout << "Neuron[" << c << "].weight = " << weight[c] << endl;
	}

	m_myIndex = myIndex;
}


// ****************** class Net ******************
class Net
{
public:
	Net(const vector<unsigned> &topology, double myWeight[100][100]);
	void feedForward(const vector<double> &inputVals);
	void getResults(vector<double> &resultVals) const;

private:
	vector<Layer> m_layers;
};

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();
	double mapminmax;

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		if (n == 0 || n == 1) {
			mapminmax = (4 - 0)*(m_layers.back()[n].getOutputVal() + 1) / (1 + 1) + 0;
		}
		if (n == 2 || n == 3) {
			mapminmax = (1 + 1)*(m_layers.back()[n].getOutputVal() + 1) / (1 + 1) - 1;
		}
		cout << "resultVals[" << n << "] = " << mapminmax << endl;
		resultVals.push_back(mapminmax);
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);
	double mapminmax;

	for (unsigned i = 0; i < inputVals.size(); ++i) {
		if (i == 0) { mapminmax = (1 + 1)*(inputVals[i] - 1250) / (2200 - 1250) - 1; }
		if (i == 1) { mapminmax = (1 + 1)*(inputVals[i] - 1300) / (2200 - 1300) - 1; }
		m_layers[0][i].setOutputVal(mapminmax);
		double input = m_layers[0][i].getOutputVal();
		cout << "input [" << i << "] = " << input << endl << endl;
 	}

	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			//cout << "feedForward - wartstwa n: " << n << endl;
			m_layers[layerNum][n].feedForward(prevLayer, layerNum);
		}
	}
}

Net::Net(const vector<unsigned> &topology, double myWeight[100][100])
{
	unsigned numLayers = topology.size();
	unsigned numWeight = 0;
	//cout << "Topology.size() = " << numLayers << endl;
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		cout << "Layer: " << layerNum << endl;
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		//cout << "numOutputs = " << numOutputs << endl;

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum, myWeight[numWeight]));
			++numWeight;
			cout << "Made a Neuron!" << endl;
		}
		cout << endl;
		m_layers.back().back().setOutputVal(1.0);
	}
}


int main()
{
	vector<unsigned> topology = { 2, 8, 4 };
	double weightTable[100][100] = { 
	{ -0.454512109914736, -2.382215605562247, 6.090167014179015, 5.700755711897299, 5.090570236291049, -4.084955404002233, -5.620588191449418, 2.611526654433665 },
	{ 8.375535749878196, 2.893159533461307, 0.673550129764684, 0.261290023014269, 0.796853049382166, 1.147367347637792, -0.614049002479692, 0.842932256950842 },
	{ -0.226100670019585, 0.805667539607018, -0.755128347652538, -0.635627631814640, 0.099757930995300, -0.730011070949842, -3.813486106911570, 2.364019101002252 },
	{ -0.063325925605889, -0.288363883771032, 0.693937887913942, 0.693945059409193 },
	{ 0.064057470869886, -0.535123160973171, -0.775805799351880, -0.775794914515106 },
	{ 0.944440069296792, -1.111596189063857, -1.471815876085993, -1.471958362195871 },
	{ -0.886123527366445, 1.558834198945437, 0.077242734810675, 0.077425107884098 },
	{ -0.568610777468602, -0.614072926632399, 2.474171082220285, 2.474135658954908 },
	{ 0.108590302466931, -0.056767826815087, 1.090749141279165, 1.090749131829550 },
	{ 0.665088047256768, -0.347975015422364, 0.721931721120446, 0.721891122466033 },
	{ 0.468727766483958, -0.750530576566738, 1.590241108777221, 1.590174954281911 },
	{ -0.181042775186012, 0.213336278405742, 0.046052586013943, 0.046073885287945 }
	};
	Net myNet(topology, weightTable);

	vector<double> input;
	vector<double> output;
	double a, b;

	while (true)
	{
		cout << endl << "IMPUT 1:" << endl;
		cin >> a;
		cout << "IMPUT 2:" << endl;
		cin >> b;
		cout << endl;
		input = { a, b };
		myNet.feedForward(input);
		myNet.getResults(output);
	}
}