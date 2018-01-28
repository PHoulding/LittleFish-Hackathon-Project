#!/usr/bin/python3
from parse import Parser
from preprocess import Preprocess
from net import NeuralNetwork

if __name__ == '__main__':
    p = Parser()
#    p.readCSV("xAPI-Edu-Data.csv")
    p.splitTrainTest("xAPI-Edu-Data.csv")
#    pre = Preprocess()
#    pre.createArrs("trainData.txt","testData.txt")
    nn = NeuralNetwork()
#    nn.train_neural_network()
    nn.test_neural_network()
