from preprocess import Preprocess
import tensorflow as tf
import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.n_nodes_hl1=50
        self.n_nodes_hl2=75
        self.n_nodes_hl3=125
        self.n_nodes_hl4=250
        self.n_nodes_hl5=100
        self.n_nodes_hl6=125
        self.n_nodes_hl7=175
        self.n_nodes_hl8=125
        self.n_nodes_hl9=75
        self.n_nodes_hl10=50

        self.n_classes=3
        self.batch_size=2

        with open("trainData.txt","r") as f:
            for i,l in enumerate(f):
                pass
            numInputs=i
        self.total_batches = int((numInputs+1)/self.batch_size)
        self.hm_epochs=500

        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')

        self.hidden_1_layer={'f_fum':self.n_nodes_hl1,\
                        'weight':tf.Variable(tf.random_normal([5,self.n_nodes_hl1])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl1]))}
        self.hidden_2_layer={'f_fum':self.n_nodes_hl2,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl1,self.n_nodes_hl2])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl2]))}
        self.hidden_3_layer={'f_fum':self.n_nodes_hl3,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl2,self.n_nodes_hl3])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl3]))}
        self.hidden_4_layer={'f_fum':self.n_nodes_hl4,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl3,self.n_nodes_hl4])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl4]))}
        self.hidden_5_layer={'f_fum':self.n_nodes_hl5,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl4,self.n_nodes_hl5])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl5]))}
        self.hidden_6_layer={'f_fum':self.n_nodes_hl6,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl5,self.n_nodes_hl6])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl6]))}
        self.hidden_7_layer={'f_fum':self.n_nodes_hl7,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl6,self.n_nodes_hl7])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl7]))}
        self.hidden_8_layer={'f_fum':self.n_nodes_hl8,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl7,self.n_nodes_hl8])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl8]))}
        self.hidden_9_layer={'f_fum':self.n_nodes_hl9,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl8,self.n_nodes_hl9])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl9]))}
        self.hidden_10_layer={'f_fum':self.n_nodes_hl10,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl9,self.n_nodes_hl10])),\
                        'bias':tf.Variable(tf.random_normal([self.n_nodes_hl10]))}
        self.output_layer={'f_fum':None,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl10,self.n_classes])),\
                        'bias':tf.Variable(tf.random_normal([self.n_classes]))}
        self.saver = tf.train.Saver()
        self.tf_log = 'tf.log'

    def neural_network_model(self,data):
        l1 = tf.add(tf.matmul(data,self.hidden_1_layer['weight']),self.hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)
        l2 = tf.add(tf.matmul(l1,self.hidden_2_layer['weight']),self.hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)
        l3 = tf.add(tf.matmul(l2,self.hidden_3_layer['weight']),self.hidden_3_layer['bias'])
        l3 = tf.nn.relu(l3)
        l4 = tf.add(tf.matmul(l3,self.hidden_4_layer['weight']),self.hidden_4_layer['bias'])
        l4 = tf.nn.relu(l4)
        l5 = tf.add(tf.matmul(l4,self.hidden_5_layer['weight']),self.hidden_5_layer['bias'])
        l5 = tf.nn.relu(l5)
        l6 = tf.add(tf.matmul(l5,self.hidden_6_layer['weight']),self.hidden_6_layer['bias'])
        l6 = tf.nn.relu(l6)
        l7 = tf.add(tf.matmul(l6,self.hidden_7_layer['weight']),self.hidden_7_layer['bias'])
        l7 = tf.nn.relu(l7)
        l8 = tf.add(tf.matmul(l7,self.hidden_8_layer['weight']),self.hidden_8_layer['bias'])
        l8 = tf.nn.relu(l8)
        l9 = tf.add(tf.matmul(l8,self.hidden_9_layer['weight']),self.hidden_9_layer['bias'])
        l9 = tf.nn.relu(l9)
        l10 = tf.add(tf.matmul(l9,self.hidden_10_layer['weight']),self.hidden_10_layer['bias'])
        l10 = tf.nn.relu(l10)

        output = tf.matmul(l10,self.output_layer['weight']+self.output_layer['bias'])
        return output

    def determineProgram(self,prog):
        if(prog=="English"):
            return 0
        elif(prog=="Spanish"):
            return 1
        elif(prog=="French"):
            return 2
        elif(prog=="Arabic"):
            return 3
        elif(prog=="IT"):
            return 4
        elif(prog=="Math"):
            return 5
        elif(prog=="Chemistry"):
            return 6
        elif(prog=="Biology"):
            return 7
        elif(prog=="Science"):
            return 8
        elif(prog=="History"):
            return 9
        elif(prog=="Quran"):
            return 10
        elif(prog=="Geology"):
            return 11
    def determineGender(self,gen):
        if(gen=="M"):
            return 0
        else:
            return 1
    def determineSemester(self,sem):
        if(sem=="F"): #First semester (Fall)
            return 0
        else: #Second semester (Winter)
            return 1
    def determineAbsent(self,abs):
        if(abs=="Above-7"):
            return 1
        else:
            return 0
    def determineFeatures(self,features):
    #    gender = self.determineGender(features[0])
    #    program = self.determineProgram(features[1])
    #    semester = self.determineSemester(features[2])
        numRaisedHand = features[3]
        numResources = features[4]
        numAnnouncements = features[5]
        numDiscussions = features[6]
        numAbsent = self.determineAbsent(features[7])
        #gender,program,semester,
        return [numRaisedHand,numResources,numAnnouncements,numDiscussions,numAbsent]
    def determineLabel(self,label):
        if(label=="L"):
            return [1,0,0]
        elif(label=="M"):
            return [0,1,0]
        else:
            return [0,0,1]
    def createFeaturesAndLabel(self,line):
        split = line.split(',')
        featurePre=[split[0],split[6],split[7],split[9],split[10],split[11],split[12],split[15]]
        labelPre=split[16][:-1]
        features = self.determineFeatures(featurePre)
        label = self.determineLabel(labelPre)
        return features,label

    def train_neural_network(self):
        prediction=-self.neural_network_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=self.y))
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(cost)
    #    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                epoch = int(open(self.tf_log,'r').read().split('\n')[-2])+1
                print("Starting:",epoch)
            except:
                epoch=1
            while(epoch<=self.hm_epochs):
                if(epoch!=1):
                    self.saver.restore(sess,"./model.ckpt")
                epoch_loss=1
                with open("trainData.txt",buffering=20000) as f:
                    batch_x=[]
                    batch_y=[]
                    batches_run=0
                    for numLine,line in enumerate(f):
                        features,label = self.createFeaturesAndLabel(line)
                        line_x = features
                        line_y = label
                        batch_x.append(line_x)
                        batch_y.append(line_y)
                        if(len(batch_x)>=self.batch_size):
                            _, c = sess.run([optimizer,cost],feed_dict={self.x:np.array(batch_x),self.y:np.array(batch_y)})
                            #optional dump output here
                            with open("dump.txt","a+") as dumpF:
                                dumpF.write(str(batch_x)+"\r\n")
                            epoch_loss+=c
                            batch_x=[]
                            batch_y=[]
                            batches_run+=1
            #                print("Batch run:",batches_run,"/",self.total_batches,"| epoch:",epoch,"| Batch loss:",c)
                self.saver.save(sess,"./model.ckpt")
                print("Epoch:",epoch,"completed out of",self.hm_epochs,"total loss",epoch_loss,"avg loss",epoch_loss/self.total_batches)
                with open(self.tf_log,"a") as f:
                    f.write(str(epoch)+"\n")
                epoch+=1

    def test_neural_network(self):
        prediction = self.neural_network_model(self.x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.hm_epochs):
                try:
                    self.saver.restore(sess,"./model.ckpt")
                except Exception as e:
                    print(str(e))
                epoch_loss=0
            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(self.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            feature_sets=[]
            labels=[]
            counter=0
            with open('testData.txt',buffering=20000) as f:
                for numLine,line in enumerate(f):
                    try:
                        features,label = self.createFeaturesAndLabel(line)
                        feature_sets.append(features)
                        labels.append(label)
                        counter+=1
                    except Exception as e:
                        print(str(e))
            print("Tested",counter,"samples")
            test_x=np.array(feature_sets)
            test_y=np.array(labels)
            print("Accuracy:",accuracy.eval({self.x:test_x,self.y:test_y}))
            output = sess.run(prediction,feed_dict={self.x:test_x})
            count=0
















            #
