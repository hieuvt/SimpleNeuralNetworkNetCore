using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Infrastructure.Function.ActivationFunction
{
    public interface IActivationFunction
    {
        double CalculateOutput(double input);
    }
}
