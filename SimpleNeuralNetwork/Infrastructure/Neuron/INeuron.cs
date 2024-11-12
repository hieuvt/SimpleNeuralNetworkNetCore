using SimpleNeuralNetwork.Infrastructure.Connection;

namespace SimpleNeuralNetwork.Infrastructure.Neuron
{
    public interface INeuron
    {
        Guid Id { get; }
        double PreviousPartialDerivate { get; set; }

        List<IConnection> Inputs { get; set; }
        List<IConnection> Outputs { get; set; }

        void AddInputNeuron(INeuron inputNeuron);
        void AddOutputNeuron(INeuron inputNeuron);
        double CalculateOutput();

        void AddInputConnection(double inputValue);
        void PushValueOnInput(double inputValue);
    }
}
