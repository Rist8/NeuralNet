# NeuralNet
Backpropagation feed forward neural network written in c++ with support of .ppm images as input


Train data format should be:

topology: 1 2 4 1
in: 1.0 0.0
out: 1.0
in: 0.0 0.0
out: 0.0


Using this example data, there would be created 4 layers with 1, 2, 4, 1 neurons respectively (first/last layers are input/output layers).
