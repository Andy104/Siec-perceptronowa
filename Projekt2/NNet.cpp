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
	cout << "tansig(x) = " << tansig << endl;
	return tansig;
}

double Neuron::purelinFunction(double x)
{
	double lin = x;
	cout << "purelin(x) = " << lin << endl;
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
			//cout << "sum = " << sum << endl;
		}
		m_outputVal = Neuron::tansigFunction(sum);
		break;
	case 2:
		for (n = 0; n < prevLayer.size(); ++n) {
			sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
			//cout << "sum = " << sum << endl;
		}
		m_outputVal = Neuron::purelinFunction(sum);
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
		if (n == 0 || n == 2) {
			mapminmax = (2 - 0)*(m_layers.back()[n].getOutputVal() + 1) / (1 + 1) + 0;
		}
		if (n == 1 || n == 3) {
			mapminmax = (1 + 1)*(m_layers.back()[n].getOutputVal() + 1) / (1 + 1) - 1;
		}
		cout << "resultVals[" << n << "] = " << mapminmax << endl;
		resultVals.push_back(mapminmax);
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); ++i) {
		double mapminmax = (1 + 1)*(inputVals[i] - 1000) / (4000 - 1000) - 1;
		m_layers[0][i].setOutputVal(mapminmax);
		double result = m_layers[0][i].getOutputVal();
		cout << "result [" << i << "] = " << result << endl;
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
	{ -2.2262353443724119, 0.240911501250247, -7.771985610902895, -0.135920811990952, -10.958964644094142, 38.361989501327564, 19.054150135051130, 6.588522811927432 },
	{ -2.3124582332364492, 11.974140101969004, 41.407328218704606, 12.065838399079153, 5.832571029618378, 1.571468312227123, -31.209524873306574, -6.167941966749249 },
	{ 6.1316912347136343, 11.81473315018301, 14.800956804396813, 11.589835002168435, 6.4383831481577252, 19.483460284520937, 15.99047038410459, -2.0871817576418819 },
	{ 0.0332948105995429, 1.03118922991245, -0.0563395212141743, 0.0838689487760870 },
	{ -4.14283514896582, -0.0680578448082310, -0.046413382601782, -0.068053620706070 },
	{ 0.004308689913784, 0.007347666907454, 0.506020624454071, 0.007348054653341 },
	{ 4.313373924496300, 0.063182512909129, 0.042335875246172, 0.063177555653699 },
	{ -0.965116013113119, -0.0005670790341832, -0.001106585549589, -0.000565684212302 },
	{ -0.004755527827990, -1.024101049166465, -0.512340546554027, -1.024149381049142 },
	{ 0.000065709634845, -0.000980268322908, -0.502208161178029, -0.000978254757323 },
	{ 1.003663346325185, 1.040947251050357, 0.523370722488140, 1.040995335965606 },
	{ 0.755945612986882, -0.017336471473698, 0.068061150656458, 0.929980775367793 }
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