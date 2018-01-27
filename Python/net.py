from preprocess import Preprocess
import tensorflow as tf
import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.n_nodes_hl1=1500
        self.n_nodes_hl2=1500
        self.n_nodes_hl3=1500
        self.n_nodes_hl4=1500

        self.n_classes=3
        self.batch_size=2

        with open("trainData.txt","r") as f:
            for i,l in enumerate(f):
                pass
            numInputs=i
        self.total_batches = int((numInputs*0.9)/self.batch_size)
        self.hm_epochs=10

        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')

        self.hidden_1_layer={'f_fum':self.n_nodes_hl1,\
                        'weight':tf.Variable(tf.random_normal([dictLength+1,self.n_nodes_hl1])),\
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
        self.output_layer={'f_fum':None,\
                        'weight':tf.Variable(tf.random_normal([self.n_nodes_hl4,self.n_classes])),\
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
            output = tf.matmul(l4,self.output_layer['weight']+self.output_layer['bias'])
            return output

        def determineLabel(label):
            if(label=="L"):
                return 0
            elif(label=="M"):
                return 1
            else:
                return 2
        def createFeaturesAndLabel(self,line):
            split = line.split(',')
            featurePre=[split[0],split[6],split[7],split[9],split[10],split[11],split[12],split[15]]
            labelPre=split[16][:-1]
            featres = determine()
            label = [determineLabel(labelPre)]
            return features,label

        def train_neural_network(self):
            prediction=-self.neural_network_model(self.x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=self.y))
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(cost)
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

















            #
