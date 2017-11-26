using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using ANN.Data;

namespace ANN.Data
{
    public class TrainingData
    {
	    public List<double> input;
        public List<double> output;



        public TrainingData()
        {
            input = new List<double>();
            output = new List<double>();
        }

        public TrainingData(List<double> _input, List<double> _output)
        {
            input = _input;
            output = _output;
        }

        public static List<TrainingData> createTrainingData(string path, int inputlength, char splitchar)
        {
            List<TrainingData> td = new List<TrainingData>();

            try
            {
                StreamReader sr = new StreamReader(path);   

                while(!sr.EndOfStream)
                {
                    string line = sr.ReadLine();

                    string[] tokens = line.Split(splitchar);
                    List<double> inputs = new List<double>();
                    List<double> outputs = new List<double>();
          
                    for(int i=0; i < inputlength; ++i)
                    {
                        inputs.Add(Convert.ToDouble(tokens[i]));
                    }
                    for (int j = inputlength; j < tokens.Count(); ++j)
                        outputs.Add(Convert.ToDouble(tokens[j]));

                    TrainingData t = new TrainingData(inputs, outputs);
                    td.Add(t);
                }
            }
            catch (FileNotFoundException fnfe)
            {
                System.Console.WriteLine("File" + path + " of TrainingData not found!");
                throw fnfe;
            }

            return td;
        }

        ///<summary>
        ///normalizes input vectors according to the zscore function. Outputs are normalized by dividing by ther maximum.
        ///</summary>
        public static void normlizeDataZScore(List<TrainingData> t, double oMax = 1.0f, double oMin = 0.0f)
        {
            double mean = 0.0f;
            double var = 0.0f;

            //normalize Input
            for (int i = 0; i < t[0].input.Count(); i++)
            {
                mean = 0;
                var = 0;
                for (int j = 0; j < t.Count(); j++)
                {
                    mean = mean + t[j].input[i];
                    var = var + t[j].input[i] * t[j].input[i];
                }
                mean = mean / t.Count();
                var = (var - t.Count() * mean * mean) / (t.Count() - 1);
                for (int j = 0; j < t.Count(); j++)
                    t[j].input[i] = (t[j].input[i] - mean) / var;
            }

            //normalize Outputs
            double Max = 0.0f;
            double Min = 0.0f;
            for (int i = 0; i < t[0].output.Count(); i++)
            {
                Max = t[0].output[i];
                Min = Max;
                for (int j = 1; j < t.Count(); j++)
                {
                    if (Max < t[j].output[i])
                        Max = t[j].output[i];
                    if (Min > t[j].output[i])
                        Min = t[j].output[i];
                }
                for (int j = 0; j < t.Count(); j++)
                    t[j].output[i] = (((t[j].output[i] - Min) * (oMax - oMin)) / (Max - Min)) + oMin;
            }
        }
        ///<summary>
        ///normalizes input vectors in range 0 to 1
        ///</summary>
        public static void normalizeData(List<TrainingData> t)
        {
            const double normLow = 0.0f;
            const double normHigh = 1.0f;
            List<double> maximum = new List<double>(10);
            List<double> minimum = new List<double>(10);

            for (int i = 0; i < maximum.Capacity; ++i)
                maximum.Add(0.0);

            for (int i = 0; i < minimum.Capacity; ++i)
                minimum.Add(0.0);

            int index = 0;
            foreach (TrainingData tdItem in t)
            {
                index = 0;
                foreach (double i in tdItem.input)
                {
                    if (maximum.ElementAt(index) < i)
                        maximum[index] = i;

                    if (minimum.ElementAt(index) > i)
                        minimum[index] = i;

                    ++index;
                }
            }

            int index2 = 0;
            for (int i = 0; i < t.Count; ++i)
            {
                index = 0;
                for (int j = 0; j < t[i].input.Count; j++)
                {
                    t[index2].input[index] = (((t.ElementAt(index2).input.ElementAt(index) - minimum.ElementAt(index)) * (normHigh - normLow)) / (maximum.ElementAt(index) - minimum.ElementAt(index))) + normLow;

                    ++index;
                }
                ++index2;
            }

            //for output vectors
            maximum.Clear();
            minimum.Clear();

            for (int i = 0; i < maximum.Capacity; ++i)
                maximum.Add(0.0);

            for (int i = 0; i < minimum.Capacity; ++i)
                minimum.Add(0.0);

            index = 0;
            foreach (TrainingData tdItem in t)
            {
                index = 0;
                foreach (double i in tdItem.output)
                {
                    if (maximum.ElementAt(index) < i)
                        maximum[index] = i;

                    if (minimum.ElementAt(index) > i)
                        minimum[index] = i;

                    ++index;
                }
            }

            index2 = 0;
            for (int i = 0; i < t.Count; ++i)
            {
                index = 0;
                for (int j = 0; j < t[i].output.Count; j++)
                {
                    t[index2].output[index] = (((t.ElementAt(index2).output.ElementAt(index) - minimum.ElementAt(index)) * (normHigh - normLow)) / (maximum.ElementAt(index) - minimum.ElementAt(index))) + normLow;

                    ++index;
                }
                ++index2;
            }

        }

    }
}
