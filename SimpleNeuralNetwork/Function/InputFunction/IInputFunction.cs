using SimpleNeuralNetwork.Infrastructure.Connection;

namespace SimpleNeuralNetwork.Infrastructure.Function.InputFunction
{
    public interface IInputFunction
    {
        double CalculateInput(List<IConnection> inputs);
    }
}
