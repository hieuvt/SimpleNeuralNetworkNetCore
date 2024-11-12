using SimpleNeuralNetwork.Infrastructure.Neuron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Infrastructure.Connection
{
    public class Connection : IConnection
    {
        internal INeuron _fromNeuron;
        internal INeuron _toNeuron;

        /// <summary>
        /// Weight of the connection.
        /// </summary>
        public double Weight { get; set; }

        /// <summary>
        /// Weight that connection had in previous itteration.
        /// Used in training process.
        /// </summary>
        public double PreviousWeight { get; set; }

        public Connection(INeuron fromNeuraon, INeuron toNeuron, double weight)
        {
            _fromNeuron = fromNeuraon;
            _toNeuron = toNeuron;

            Weight = weight;
            PreviousWeight = 0;
        }

        public static Random rand = new Random();

        public Connection(INeuron fromNeuraon, INeuron toNeuron)
        {
            _fromNeuron = fromNeuraon;
            _toNeuron = toNeuron;

            // var tmpRandom = new Random();
            // Weight = tmpRandom.NextDouble();
            Weight = rand.NextDouble();
            PreviousWeight = 0;
        }

        /// <summary>
        /// Get output value of the connection.
        /// </summary>
        /// <returns>
        /// Output value of the connection.
        /// </returns>
        public double GetOutput()
        {
            return _fromNeuron.CalculateOutput();
        }

        /// <summary>
        /// Checks if Neuron has a certain number as an input neuron.
        /// </summary>
        /// <param name="fromNeuronId">Neuron Id.</param>
        /// <returns>
        /// True - if the neuron is the input of the connection.
        /// False - if the neuron is not the input of the connection. 
        /// </returns>
        public bool IsFromNeuron(Guid fromNeuronId)
        {
            return _fromNeuron.Id.Equals(fromNeuronId);
        }

        /// <summary>
        /// Update weight.
        /// </summary>
        /// <param name="learningRate">Chossen learning rate.</param>
        /// <param name="delta">Calculated difference for which weight of the connection needs to be modified.</param>
        public void UpdateWeight(double learningRate, double delta)
        {
            PreviousWeight = Weight;
            Weight += learningRate * delta;
        }
    }
}
