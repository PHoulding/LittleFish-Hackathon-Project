#!/usr/bin/python3
from parse import Parser
from preprocess import Preprocess

if __name__ == '__main__':
    p = Parser()
#    p.readCSV("xAPI-Edu-Data.csv")
#    p.splitTrainTest("xAPI-Edu-Data.csv")
#    pre = Preprocess()
#    pre.createArrs("trainData.txt","testData.txt")
    with open("trainData.txt",buffering=20000) as f:
        for numLine,line in enumerate(f):
            split = line.split(',')
            featurePre=[split[0],split[6],split[7],split[9],split[10],split[11],split[12],split[15]]
            labelPre=split[16][:-1]

            
            print(featurePre," ",labelPre)
