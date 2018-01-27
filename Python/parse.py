import csv
import random
import re

class Parser():
    def __init__(self):
        self.rows=[]
    def readCSV(self,csvFile):
        with open(csvFile,newline='') as csvFile:
            reader = csv.reader(csvFile,delimiter=',',quotechar='|')
            for row in reader:
                self.rows.append(row)
    def writeTrainTest(self,trainData,testData):
        with open("trainData.txt",'w+') as f:
            for line in trainData:
                f.write(line)
        with open("testData.txt",'w+') as f:
            for line in testData:
                f.write(line)
    def printCSV(self):
        for row in self.rows:
            print(row)
    def checkLast(self):
        high=0
        low=0
        mid=0
        for row in self.rows:
            if(row[-1]=="H"):
                high+=1
            if(row[-1]=="M"):
                mid+=1
            if(row[-1]=="L"):
                low+=1
        print("High=",high,"Mid=",mid,"Low=",low)
    def splitTrainTest(self,inputFile):
        inputData=[]
        trainData=[]
        testData=[]
        with open(inputFile,"r") as f:
            for i,l in enumerate(f):
                if(i>0):
                    inputData.append(l)
        random.shuffle(inputData)
        split=int(len(inputData)*0.9)
        trainData=inputData[:split]
        testData=inputData[split:]
        self.writeTrainTest(trainData,testData)











            #
