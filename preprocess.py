import String

class Preprocess():
    def createArrs(self,trainData,testData):
        with open(trainData,"r"):
            for i,l in enumerate(f):
                split = l.split(",")
                
