

class Preprocess():
    def __init__(self):
        self.trainArrs=[]
        self.testArrs=[]
        self.arr=[]
    def createArrs(self,trainData,testData):
        with open(trainData,"r") as f:
            for i,l in enumerate(f):
                split = l.split(",")
                lineArray=[split[0],split[6],split[7],split[9],split[10],split[11],split[12],split[15],split[16][:-1]]
                self.trainArrs.append(lineArray)
        with open(testData,"r") as f:
            for i,l in enumerate(f):
                split = l.split(",")
                lineArray=[split[0],split[6],split[7],split[9],split[10],split[11],split[12],split[15],split[16][:-1]]
                self.testArrs.append(lineArray)
    def createArr(self,data):
        with open(data,"r") as f:
            for i,l in enumerate(f):
                split = l.split(",")
                lineArray=[split[0],split[6],split[7],split[9],split[10],split[11],split[12],split[15],split[16][:-1]]
                self.arr.append(lineArray)
