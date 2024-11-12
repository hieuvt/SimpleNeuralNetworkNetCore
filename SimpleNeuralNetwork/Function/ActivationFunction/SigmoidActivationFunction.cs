﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Infrastructure.Function.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private double _coeficient;

        public SigmoidActivationFunction(double coeficient)
        {
            _coeficient = coeficient;
        }

        public double CalculateOutput(double input)
        {
            return 1 / (1 + Math.Exp(-input * _coeficient));
        }
    }
}
