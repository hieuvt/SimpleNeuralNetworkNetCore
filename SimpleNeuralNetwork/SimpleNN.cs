using SimpleNeuralNetwork.Data;
using SimpleNeuralNetwork.Infrastructure.Function.ActivationFunction;
using SimpleNeuralNetwork.Infrastructure.Function.InputFunction;
using SimpleNeuralNetwork.Infrastructure.NeuronLayer;

namespace SimpleNeuralNetwork
{
    public class SimpleNN
    {
        private NeuralLayerFactory _layerFactory;

        internal List<NeuralLayer> _layers;
        internal double _learningRate;
        internal double[][] _expectedResult;

        /// <summary>
        /// Constructor of the Neural Network.
        /// Note:
        /// Initialy input layer with defined number of inputs will be created.
        /// </summary>
        /// <param name="numberOfInputNeurons">
        /// Number of neurons in input layer.
        /// </param>
        public SimpleNN(int numberOfInputNeurons)
        {
            _layers = new List<NeuralLayer>();
            _layerFactory = new NeuralLayerFactory();

            // Create input layer that will collect inputs.
            CreateInputLayer(numberOfInputNeurons);

            _learningRate = 20; //2.95;
        }

        /// <summary>
        /// Add layer to the neural network.
        /// Layer will automatically be added as the output layer to the last layer in the neural network.
        /// </summary>
        public void AddLayer(NeuralLayer newLayer)
        {
            if (_layers.Any())
            {
                var lastLayer = _layers.Last();
                newLayer.ConnectLayers(lastLayer);
            }

            _layers.Add(newLayer);
        }

        /// <summary>
        /// Push input values to the neural network.
        /// </summary>
        public void PushInputValues(double[] inputs)
        {
            _layers.First().Neurons.ForEach(x => x.PushValueOnInput(inputs[_layers.First().Neurons.IndexOf(x)]));
        }

        /// <summary>
        /// Set expected values for the outputs.
        /// </summary>
        //public void PushExpectedValues(double[][] expectedOutputs)
        //{
        //    _expectedResult = expectedOutputs;
        //}

        /// <summary>
        /// Calculate output of the neural network.
        /// </summary>
        /// <returns></returns>
        public List<double> GetOutput()
        {
            var returnValue = new List<double>();

            _layers.Last().Neurons.ForEach(neuron =>
            {
                returnValue.Add(neuron.CalculateOutput());
            });

            return returnValue;
        }

        /// <summary>
        /// Train neural network.
        /// </summary>
        /// <param name="inputs">Input values.</param>
        /// <param name="numberOfEpochs">Number of epochs.</param>
        public void Train(List<TrainingModel> trainingDatas, int numberOfEpochs)
        {
            double totalError = 0;

            for (int i = 0; i < numberOfEpochs; i++)
            {
                foreach (var trainingData in trainingDatas)
                {
                    foreach (var input in trainingData.Inputs)
                    {
                        PushInputValues(input);

                        var outputs = new List<double>();

                        // Get outputs.
                        _layers.Last().Neurons.ForEach(x =>
                        {
                            outputs.Add(x.CalculateOutput());
                        });

                        // Calculate error by summing errors on all output neurons.
                        totalError = CalculateTotalError(outputs, trainingData.LabeledOutputs);
                        HandleOutputLayer(trainingData.LabeledOutputs);
                        HandleHiddenLayers();
                    }


                }
            }
        }

        /// <summary>
        /// Hellper function that creates input layer of the neural network.
        /// </summary>
        private void CreateInputLayer(int numberOfInputNeurons)
        {
            var inputLayer = _layerFactory.CreateNeuralLayer(numberOfInputNeurons, new RectifiedActivationFuncion(), new WeightedSumFunction());
            inputLayer.Neurons.ForEach(x => x.AddInputConnection(0));
            this.AddLayer(inputLayer);
        }

        /// <summary>
        /// Hellper function that calculates total error of the neural network.
        /// </summary>
        private double CalculateTotalError(List<double> outputs, double[] labeledOutputs)
        {
            double totalError = 0;

            outputs.ForEach(output =>
            {
                var error = Math.Pow(output - labeledOutputs[outputs.IndexOf(output)], 2);
                totalError += error;
            });

            return totalError;
        }

        /// <summary>
        /// Hellper function that runs backpropagation algorithm on the output layer of the network.
        /// </summary>
        /// <param name="row">
        /// Input/Expected output row.
        /// </param>
        private void HandleOutputLayer(double[] labeledResult)
        {
            _layers.Last().Neurons.ForEach(neuron =>
            {
                neuron.Inputs.ForEach(connection =>
                {
                    var output = neuron.CalculateOutput();
                    var netInput = connection.GetOutput();

                    var expectedOutput = labeledResult[_layers.Last().Neurons.IndexOf(neuron)];

                    var nodeDelta = (expectedOutput - output) * output * (1 - output);
                    //var delta = -1 * netInput * nodeDelta;
                    var delta = netInput * nodeDelta;

                    connection.UpdateWeight(_learningRate, delta);

                    neuron.PreviousPartialDerivate = nodeDelta;
                });
            });
        }

        /// <summary>
        /// Hellper function that runs backpropagation algorithm on the hidden layer of the network.
        /// </summary>
        /// <param name="row">
        /// Input/Expected output row.
        /// </param>
        private void HandleHiddenLayers()
        {
            for (int k = _layers.Count - 2; k > 0; k--)
            {
                _layers[k].Neurons.ForEach(neuron =>
                {
                    neuron.Inputs.ForEach(connection =>
                    {
                        var output = neuron.CalculateOutput();
                        var netInput = connection.GetOutput();
                        double sumPartial = 0;

                        _layers[k + 1].Neurons
                        .ForEach(outputNeuron =>
                        {
                            outputNeuron.Inputs.Where(i => i.IsFromNeuron(neuron.Id))
                            .ToList()
                            .ForEach(outConnection =>
                            {
                                sumPartial += outConnection.PreviousWeight * outputNeuron.PreviousPartialDerivate;
                            });
                        });

                        //var delta = -1 * netInput * sumPartial * output * (1 - output);
                        var delta = netInput * sumPartial * output * (1 - output);
                        connection.UpdateWeight(_learningRate, delta);
                    });
                });
            }
        }

        public double[] DivideDoubleArray(double[] arr, double divideBy)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] /= divideBy;
            }
            return arr;
        }
    }
}
