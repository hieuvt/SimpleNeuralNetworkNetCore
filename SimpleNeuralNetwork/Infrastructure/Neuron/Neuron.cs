using SimpleNeuralNetwork.Infrastructure.Connection;
using SimpleNeuralNetwork.Infrastructure.Function.ActivationFunction;
using SimpleNeuralNetwork.Infrastructure.Function.InputFunction;

namespace SimpleNeuralNetwork.Infrastructure.Neuron
{
    public class Neuron : INeuron
    {
        private IActivationFunction _activationFunction;
        private IInputFunction _inputFunction;

        /// <summary>
        /// Input connections of the neuron.
        /// </summary>
        public List<IConnection> Inputs { get; set; }

        /// <summary>
        /// Output connections of the neuron.
        /// </summary>
        public List<IConnection> Outputs { get; set; }

        public Guid Id { get; private set; }

        /// <summary>
        /// Calculated partial derivate in previous iteration of training process.
        /// </summary>
        public double PreviousPartialDerivate { get; set; }

        public Neuron(IActivationFunction activationFunction, IInputFunction inputFunction)
        {
            Id = Guid.NewGuid();
            Inputs = new List<IConnection>();
            Outputs = new List<IConnection>();

            _activationFunction = activationFunction;
            _inputFunction = inputFunction;
        }

        /// <summary>
        /// Connect two neurons. 
        /// This neuron is the output neuron of the connection.
        /// </summary>
        /// <param name="inputNeuron">Neuron that will be input neuron of the newly created connection.</param>
        public void AddInputNeuron(INeuron inputNeuron)
        {
            var connection = new SimpleNeuralNetwork.Infrastructure.Connection.Connection(inputNeuron, this);
            Inputs.Add(connection);
            inputNeuron.Outputs.Add(connection);
        }

        /// <summary>
        /// Connect two neurons. 
        /// This neuron is the input neuron of the connection.
        /// </summary>
        /// <param name="outputNeuron">Neuron that will be output neuron of the newly created connection.</param>
        public void AddOutputNeuron(INeuron outputNeuron)
        {
            var connection = new SimpleNeuralNetwork.Infrastructure.Connection.Connection(this, outputNeuron);
            Outputs.Add(connection);
            outputNeuron.Inputs.Add(connection);
        }

        /// <summary>
        /// Calculate output value of the neuron.
        /// </summary>
        /// <returns>
        /// Output of the neuron.
        /// </returns>
        public double CalculateOutput()
        {
            return _activationFunction.CalculateOutput(_inputFunction.CalculateInput(Inputs));
        }

        /// <summary>
        /// Input Layer neurons just receive input values.
        /// For this they need to have connections.
        /// This function adds this kind of connection to the neuron.
        /// </summary>
        /// <param name="inputValue">
        /// Initial value that will be "pushed" as an input to connection.
        /// </param>
        public void AddInputConnection(double inputValue)
        {
            var inputConnection = new InputConnection(this, inputValue);
            Inputs.Add(inputConnection);
        }

        /// <summary>
        /// Sets new value on the input connections.
        /// </summary>
        /// <param name="inputValue">
        /// New value that will be "pushed" as an input to connection.
        /// </param>
        public void PushValueOnInput(double inputValue)
        {
            ((InputConnection)Inputs.First()).Output = inputValue;
        }
    }
}
