using SimpleNeuralNetwork.Infrastructure.Connection;

namespace SimpleNeuralNetwork.Infrastructure.Function.InputFunction
{
    public class WeightedSumFunction : IInputFunction
    {
        public double CalculateInput(List<IConnection> inputs)
        {
            return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
        }
    }
}
