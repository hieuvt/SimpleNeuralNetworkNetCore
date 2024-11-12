using SimpleNeuralNetwork.Infrastructure.Function.ActivationFunction;
using SimpleNeuralNetwork.Infrastructure.Function.InputFunction;

namespace SimpleNeuralNetwork.Infrastructure.NeuronLayer
{
    public class NeuralLayerFactory
    {
        public NeuralLayer CreateNeuralLayer(int numberOfNeurons, IActivationFunction activationFunction, IInputFunction inputFunction)
        {
            var layer = new NeuralLayer();

            for (int i = 0; i < numberOfNeurons; i++)
            {
                var neuron = new SimpleNeuralNetwork.Infrastructure.Neuron.Neuron(activationFunction, inputFunction);
                layer.Neurons.Add(neuron);
            }

            return layer;
        }
    }
}
