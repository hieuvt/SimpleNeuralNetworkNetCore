//https://rodansotto.github.io/tech-blog/2022/08/07/how-i-trained-my-csharp-neural-network.html
// So our input data represents a 5x7 1-bit image that contains a number from 0 to 9
// To save space, the input data is an integer array of 7 corresponding to the 7 rows of 5 pixels in that 5x7 image
// Each row contains the decimal value of the 5 pixels in that row
//  e.g. if row contains these 5 pixels [0 1 1 1 0], then the decimal value will be 14

// create neural network with 7 neurons in the input layer
using SimpleNeuralNetwork;
using SimpleNeuralNetwork.Data;
using SimpleNeuralNetwork.Infrastructure.Function.ActivationFunction;
using SimpleNeuralNetwork.Infrastructure.Function.InputFunction;
using SimpleNeuralNetwork.Infrastructure.NeuronLayer;

var network = new SimpleNN(7); // <- the input layer will be created here

// add 2 layers, each with a number of neurons and activation and input functions to use
var layerFactory = new NeuralLayerFactory();
network.AddLayer(layerFactory.CreateNeuralLayer(4, new SigmoidActivationFunction(1), new WeightedSumFunction())); // <- this is the hidden layer
network.AddLayer(layerFactory.CreateNeuralLayer(1, new SigmoidActivationFunction(1), new WeightedSumFunction())); // <- this is the output layer

// define expected outputs
// if you notice I'm dividing the values by 10 because apparently this neural network does not work if they are not between 0 and 1 (including 0 and 1)
//network.PushExpectedValues(
//    new double[][] {
//                network.DivideDoubleArray(new double[] { 1 }, 10),
//                network.DivideDoubleArray(new double[] { 2 }, 10),
//                network.DivideDoubleArray(new double[] { 3 }, 10),
//                network.DivideDoubleArray(new double[] { 4 }, 10),
//                network.DivideDoubleArray(new double[] { 5 }, 10),
//                network.DivideDoubleArray(new double[] { 6 }, 10),
//                network.DivideDoubleArray(new double[] { 7 }, 10),
//                network.DivideDoubleArray(new double[] { 8 }, 10),
//                network.DivideDoubleArray(new double[] { 9 }, 10),
//                network.DivideDoubleArray(new double[] { 0 }, 10),
//    });

//// train with the input training set and number of epochs (or times) it will run the training set through the network
//// again I'm dividing the values so they stay within 0 and 1, otherwise it will not work
//// I'm dividing by 31 as the maximum value a row of pixel in our 5x7 image can have is [1 1 1 1 1] = 31 in decimal
Console.WriteLine("Training...");
//network.Train(
//    new double[][] {
//                network.DivideDoubleArray(new double[] { 4, 12, 4, 4, 4, 4, 4 }, 31),
//                network.DivideDoubleArray(new double[] { 6, 9, 1, 2, 4, 8, 15 }, 31),
//                network.DivideDoubleArray(new double[] { 6, 9, 1, 6, 1, 9, 6 }, 31),
//                network.DivideDoubleArray(new double[] { 3, 5, 5, 9, 15, 1, 1 }, 31),
//                network.DivideDoubleArray(new double[] { 14, 8, 8, 14, 1, 1, 14 }, 31),
//                network.DivideDoubleArray(new double[] { 6, 9, 16, 14, 17, 17, 14 }, 31),
//                network.DivideDoubleArray(new double[] { 14, 1, 1, 2, 2, 4, 4 }, 31),
//                network.DivideDoubleArray(new double[] { 6, 9, 9, 6, 9, 9, 6 }, 31),
//                network.DivideDoubleArray(new double[] { 6, 9, 9, 9, 7, 1, 6 }, 31),
//                network.DivideDoubleArray(new double[] { 6, 9, 9, 9, 9, 9, 6 }, 31),
//    }, 10000);

var trainingDatas = new List<TrainingModel>();
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 4, 12, 4, 4, 4, 4, 4 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 1 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 6, 9, 1, 2, 4, 8, 15 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 2 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 6, 9, 1, 6, 1, 9, 6 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 3 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 3, 5, 5, 9, 15, 1, 1 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 4 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 14, 8, 8, 14, 1, 1, 14 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 5 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 6, 9, 16, 14, 17, 17, 14 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 6 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 6, 9, 9, 6, 9, 9, 6 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 8 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 6, 9, 9, 9, 7, 1, 6 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 9 }, 10)
});
trainingDatas.Add(new TrainingModel()
{
    Inputs = new List<double[]> { network.DivideDoubleArray(new double[] { 6, 9, 9, 9, 9, 9, 6 }, 31) },
    LabeledOutputs = network.DivideDoubleArray(new double[] { 0 }, 10)
});

network.Train(trainingDatas, 10000);

// and now to test how good the neural network was trained :)
Console.WriteLine("Testing...");
network.PushInputValues(network.DivideDoubleArray(new double[] { 4, 12, 4, 4, 4, 4, 4 }, 31));
var outputs = network.GetOutput();
Console.WriteLine($"Input: 1; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 6, 9, 1, 2, 4, 8, 15 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 2; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 6, 9, 1, 6, 1, 9, 6 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 3; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 3, 5, 5, 9, 15, 1, 1 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 4; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 14, 8, 8, 14, 1, 1, 14 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 5; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 6, 9, 16, 14, 17, 17, 14 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 6; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 14, 1, 1, 2, 2, 4, 4 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 7; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 6, 9, 9, 6, 9, 9, 6 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 8; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 6, 9, 9, 9, 7, 1, 6 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 9; Output: {string.Join(',', outputs)}");

network.PushInputValues(network.DivideDoubleArray(new double[] { 6, 9, 9, 9, 9, 9, 6 }, 31));
outputs = network.GetOutput();
Console.WriteLine($"Input: 0; Output: {string.Join(',', outputs)}");