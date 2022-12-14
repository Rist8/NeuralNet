#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

//Process a binary PPM file
#include <vector>
#include <string>

#ifndef PPM_H
#define PPM_H

class ppm {
    void init();
    //info about the PPM file (height and width)
    unsigned int nr_lines;
    unsigned int nr_columns;

public:
    //arrays for storing the R,G,B values
    std::vector<unsigned char> r;
    std::vector<unsigned char> g;
    std::vector<unsigned char> b;
    //
    unsigned int height;
    unsigned int width;
    unsigned int max_col_val;
    //total number of elements (pixels)
    unsigned int size;

    ppm();
    //create a PPM object and fill it with data stored in fname 
    ppm(const std::string& fname);
    //create an "epmty" PPM image with a given width and height;the R,G,B arrays are filled with zeros
    ppm(const unsigned int _width, const unsigned int _height);
    //read the PPM image from fname
    void read(const std::string& fname);
    //write the PPM image in fname
    void write(const std::string& fname);
};

#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

//init with default values

void ppm::init() {
    width = 0;
    height = 0;
    max_col_val = 255;
}

//create a PPM object

ppm::ppm() {
    init();
}

//create a PPM object and fill it with data stored in fname 

ppm::ppm(const std::string& fname) {
    init();
    read(fname);
}

//create an "epmty" PPM image with a given width and height;the R,G,B arrays are filled with zeros

ppm::ppm(const unsigned int _width, const unsigned int _height) {
    init();
    width = _width;
    height = _height;
    nr_lines = height;
    nr_columns = width;
    size = width * height;

    // fill r, g and b with 0
    r.resize(size);
    g.resize(size);
    b.resize(size);
}

//read the PPM image from fname

void ppm::read(const std::string& fname) {
    std::ifstream inp(fname.c_str(), std::ios::in | std::ios::binary);
    if (inp.is_open()) {
        std::string line;
        std::getline(inp, line);
        if (line != "P6") {
            std::cout << "Error. Unrecognized file format." << std::endl;
            return;
        }
        std::getline(inp, line);
        while (line[0] == '#') {
            std::getline(inp, line);
        }
        std::stringstream dimensions(line);

        try {
            dimensions >> width;
            dimensions >> height;
            nr_lines = height;
            nr_columns = width;
        }
        catch (std::exception& e) {
            std::cout << "Header file format error. " << e.what() << std::endl;
            return;
        }

        std::getline(inp, line);
        std::stringstream max_val(line);
        try {
            max_val >> max_col_val;
        }
        catch (std::exception& e) {
            std::cout << "Header file format error. " << e.what() << std::endl;
            return;
        }

        size = width * height;

        r.reserve(size);
        g.reserve(size);
        b.reserve(size);

        char aux;
        for (unsigned int i = 0; i < size; ++i) {
            inp.read(&aux, 1);
            r[i] = (unsigned char)aux;
            inp.read(&aux, 1);
            g[i] = (unsigned char)aux;
            inp.read(&aux, 1);
            b[i] = (unsigned char)aux;
        }
    }
    else {
        std::cout << "Error. Unable to open " << fname << std::endl;
    }
    inp.close();
}

//write the PPM image in fname

void ppm::write(const std::string& fname) {
    std::ofstream inp(fname.c_str(), std::ios::out | std::ios::binary);
    if (inp.is_open()) {

        inp << "P6\n";
        inp << width;
        inp << " ";
        inp << height << "\n";
        inp << max_col_val << "\n";

        char aux;
        for (unsigned int i = 0; i < size; ++i) {
            aux = (char)r[i];
            inp.write(&aux, 1);
            aux = (char)g[i];
            inp.write(&aux, 1);
            aux = (char)b[i];
            inp.write(&aux, 1);
        }
    }
    else {
        std::cout << "Error. Unable to open " << fname << std::endl;
    }
    inp.close();
}


class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned>& topology);
    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<long double>& inputVals);
    unsigned getTargetOutputs(vector<long double>& targetOutputVals);
private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned>& topology)
{
    string line;
    string label;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0)
        abort();
    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<long double>& inputVals)
{
    inputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        long double oneValue;
        while (ss >> oneValue)
            inputVals.push_back(oneValue);
    } else
    if (label.compare("ppm:") == 0) {
        string oneValue;
        ss >> oneValue;
        oneValue += ".ppm";
        ppm img(oneValue);
        unsigned kk = 0;
        for(int i = 0; i < 64; ++i)
            for (int j = 0; j < 64; ++j) {
                inputVals.push_back(img.r[kk]);
                ++kk;
            }
    }
    return unsigned(inputVals.size());
}

unsigned TrainingData::getTargetOutputs(vector<long double>& targetOutputVals)
{
    targetOutputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        long double oneValue;
        while (ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }
    return unsigned(targetOutputVals.size());
}

struct Connection
{
    long double weight;
    long double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(long double val) { m_outputVal = val; }
    long double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer& prevLayer);
    void calcOutputGradients(long double targetVal);
    void calcHiddenGradients(const Layer& nextLayer);
    void updateInputWeights(Layer& prevLayer);
    vector<Connection> getOutputWeights(void) { return m_outputWeights; }
    void setOutputWeights(vector<Connection> x) { m_outputWeights = x; }
    long double getGradient(void) { return m_gradient; }
    void setGradient(long double x) { m_gradient = x; }

private:
    static long double eta;
    static long double alpha;
    static long double transferFunction(long double x);
    static long double transferFunctionDerivative(long double x);
    static long double randomWeight(void) { return rand() / long double(RAND_MAX); }
    long double sumDOW(const Layer& nextLayer) const;
    long double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    long double m_gradient;
};

long double Neuron::eta = 0.05;
long double Neuron::alpha = 0;


void Neuron::updateInputWeights(Layer& prevLayer)
{

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron& neuron = prevLayer[n];
        long double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        long double newDeltaWeight = 
            // Individual input, magnified by the gradient and train rate:
            eta
            * neuron.getOutputVal()
            * m_gradient
            // Also add momentum = a fraction of the previous delta weight;
            + alpha
            * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

long double Neuron::sumDOW(const Layer& nextLayer) const
{
    long double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size(); ++n)
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    return sum;
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
    long double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(long double targetVal)
{
    long double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

long double Neuron::transferFunction(long double x)
{
    return tanh(x);
}

long double Neuron::transferFunctionDerivative(long double x)
{
    return 1.0 - tanh(x) * tanh(x);
}

void Neuron::feedForward(const Layer& prevLayer)
{
    long double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}


class Net
{
public:
    Net(const vector<unsigned>& topology);
    Net(ifstream *fin, vector<unsigned>* topology);
    void feedForward(const vector<long double>& inputVals);
    void backProp(const vector<long double>& targetVals);
    void getResults(vector<long double>& resultVals) const;
    long double getRecentAverageError(void) const { return m_recentAverageError; }
    void printNeuralNet(const string filename, const vector<unsigned>& topology);
private:
    // m_layers[layerNum][neuronNum]
    vector<Layer> m_layers;
    long double m_error;
    long double m_recentAverageError;
    static long double m_recentAverageSmoothingFactor;
};


long double Net::m_recentAverageSmoothingFactor = 100.0;

void Net::printNeuralNet(const string filename, const vector<unsigned>& topology)
{
    if(filename != "BestNeuralNet.txt")
        cout << endl << "Saving neural net...";
    else
        cout << endl << "Saving as best neural net...";
    ofstream fout(filename);
    fout.precision(17);
    unsigned numLayers = unsigned(topology.size());
    fout << numLayers << '\n';
    for (int i = 0; i < topology.size(); ++i)
        fout << topology[i] << ' ';
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {

        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[static_cast<std::vector<unsigned int, std::allocator<unsigned int>>::size_type>(layerNum) + 1];
        fout << '\n' << numOutputs << '\n';
        for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
            fout << m_layers[layerNum][neuronNum].getOutputVal() << ' ' << neuronNum << ' ' << m_layers[layerNum][neuronNum].getGradient() << '\n';
            vector<Connection> outputWeights = m_layers[layerNum][neuronNum].getOutputWeights();
            for (unsigned i = 0; i < numOutputs; ++i)
                fout << outputWeights[i].weight << ' ' << outputWeights[i].deltaWeight << '\n';
        }
    }
    fout.close();
    if (filename != "BestNeuralNet.txt")
        cout << endl << "Saved succesfully";
    else
        cout << endl << "Saved succesfully";
}


void Net::getResults(vector<long double>& resultVals) const
{
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size(); ++n)
        resultVals.push_back(m_layers.back()[n].getOutputVal());
}

void Net::backProp(const vector<long double>& targetVals)
{
    Layer& outputLayer = m_layers.back();
    m_error = 0.0;
    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        long double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size(); // get average error squared
    m_error = sqrt(m_error); // RMS
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);
    for (unsigned n = 0; n < outputLayer.size(); ++n)
        outputLayer[n].calcOutputGradients(targetVals[n]);
    for (unsigned layerNum = unsigned(m_layers.size()) - 2; layerNum > 0; --layerNum) {
        Layer& hiddenLayer = m_layers[layerNum];
        Layer& nextLayer = m_layers[static_cast<std::vector<Layer, std::allocator<Layer>>::size_type>(layerNum) + 1];
        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
            hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
    for (unsigned layerNum = unsigned(m_layers.size()) - 1; layerNum > 0; --layerNum) {
        Layer& layer = m_layers[layerNum];
        Layer& prevLayer = m_layers[static_cast<std::vector<Layer, std::allocator<Layer>>::size_type>(layerNum) - 1];
        for (unsigned n = 0; n < layer.size(); ++n)
            layer[n].updateInputWeights(prevLayer);
    }
}

void Net::feedForward(const vector<long double>& inputVals)
{
    assert(inputVals.size() == m_layers[0].size());
    for (unsigned i = 0; i < inputVals.size(); ++i)
        m_layers[0][i].setOutputVal(inputVals[i]);
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer& prevLayer = m_layers[static_cast<std::vector<Layer, std::allocator<Layer>>::size_type>(layerNum) - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size(); ++n)
            m_layers[layerNum][n].feedForward(prevLayer);
    }
}

Net::Net(const vector<unsigned>& topology)
{
    m_recentAverageError = 0;
    m_error = 0;
    unsigned numLayers = unsigned(topology.size());
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[static_cast<std::vector<unsigned int, std::allocator<unsigned int>>::size_type>(layerNum) + 1];
        for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum)
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        m_layers.back().back().setOutputVal(1.0);
    }
}

Net::Net(ifstream *fin, vector<unsigned>* topology)
{
    m_recentAverageError = 0;
    m_error = 0;
    unsigned numLayers;
    (*topology).clear();
    *fin >> numLayers;
    for (unsigned i = 0; i < numLayers; ++i) {
        unsigned t = 0;
        *fin >> t;
        (*topology).push_back(t);
    }
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = 0;
        *fin >> numOutputs;
        for (unsigned neuronNum = 0; neuronNum < (*topology)[layerNum]; ++neuronNum) {
            long double val, gradient;
            unsigned num;
            *fin >> val >> num >> gradient;
            m_layers.back().push_back(Neuron(numOutputs, num));
            m_layers.back().back().setGradient(gradient);
            m_layers.back().back().setOutputVal(val);
            vector<Connection> outputWeights;
            Connection a = {0};
            for (unsigned i = 0; i < numOutputs; ++i) {
                *fin >> a.weight >> a.deltaWeight;
                outputWeights.push_back(a);
            }
            m_layers.back().back().setOutputWeights(outputWeights);
        }
    }
}


void showVectorVals(string label, vector<long double>& v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
        cout << v[i] << " ";
    cout << endl;
}

unsigned showVectorMaxVals(string label, vector<long double>& v, bool pri)
{
    long double max = -2;
    unsigned num = 0;
    if(pri)
        cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
        if (v[i] >= max) {
            max = v[i];
            num = i;
        }
    if(pri)
        cout << num << endl;
    return num;
}



int main(int argc, char** argv)
{
    bool train = 0;
    bool _PrintRes = 1;
    //if (argc >= 2)
      //  train = atoi(argv[1]);
    //if (argc == 3)
      //  _PrintRes = atoi(argv[2]);
    long double bestError = 999;
    for (int i = 0; i < 10000; ++i) {
        unsigned inARow = 0;
        long double AvgError = 0;
        TrainingData trainData("trainingData.txt");
        vector<unsigned> topology;
        trainData.getTopology(topology);
        Net myNet(topology);
        ifstream fin;
        if(train)
            fin.open("neuralNet.txt");
        else
            fin.open("BestNeuralNet.txt");
        fin.precision(17);
        if (fin.is_open()) {
            Net myNet1(&fin, &topology);
            myNet = myNet1;
        }
        fin.close();
        vector<long double> inputVals, targetVals, resultVals;
        unsigned trainingPass = 0;
        while (!trainData.isEof()) {
            ++trainingPass;
            if (_PrintRes)
                cout << endl << "Number " << (trainingPass - 1) % 10 << '\n';
            // Get new input data and feed it forward:
            if (trainData.getNextInputs(inputVals) != topology[0])
                break;
            //if(_PrintRes)
                //showVectorVals(": Inputs:", inputVals);
            myNet.feedForward(inputVals);
            // Collect the net's actual output results:
            myNet.getResults(resultVals);
            unsigned out = 0;
            out = showVectorMaxVals("Outputs:", resultVals, _PrintRes);
            // Train the net what the outputs should have been:
            trainData.getTargetOutputs(targetVals);
            showVectorMaxVals("Targets:", targetVals, _PrintRes);
            if (targetVals[out] == 1.0)
                ++inARow;
            if (train) {
                assert(targetVals.size() == topology.back());
                myNet.backProp(targetVals);
            }
            // Report how well the training is working, average over recent samples:
            if (_PrintRes && train)
                cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
            AvgError += myNet.getRecentAverageError();
        }
        AvgError /= trainingPass;
        cout << endl << i;
        if(train)
            myNet.printNeuralNet("neuralNet.txt", topology);
        if (bestError >= AvgError && train){
            myNet.printNeuralNet("BestNeuralNet.txt", topology);
            bestError = AvgError;
        }
        if(train)
            cout << endl << "Data set recent average error: " << AvgError << endl;
        cout << "Data set accuracy: " << long double(long double(long double(inARow)/long double(trainingPass)) * 100) << '%' << endl;
    }
    cout << endl << "Done" << endl;
}
