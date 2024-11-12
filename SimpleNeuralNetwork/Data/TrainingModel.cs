using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Data
{
    public class TrainingModel
    {
        public List<double[]> Inputs { get; set; }
        public double[] LabeledOutputs { get; set; }
    }
}
