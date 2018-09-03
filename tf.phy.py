# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:12:16 2018

@author: tc
"""

import tensorflow as tf
import numpy as np
import keras 
import matplotlib.pyplot as plt

n = 100000

g = 9.8
pi = np.pi
vd = 10
vu = 50
v0 = vd+(vu-vd)*np.random.rand(n)
V0 = np.zeros((n,1),dtype=float)
theta = 0.5*pi*np.random.rand(n)
Theta = np.zeros((n,1),dtype=float)
sin_theta = np.sin(theta)
y_max = np.power(vu,2)*np.power(np.sin(pi/2),2)/(2*g)
#sin_Theta = np.zeros((n,1),dtype=float)
#np.random.shuffle(V0)    # randomize the data
#y = np.power(v0,2) * np.power(sin_theta)/g
#y = y.astype('float32')
Y = np.zeros((n,1))
y = np.zeros((n,1),dtype=float)

# normalize y v0 theta
for i in range(n):
    y[i] = np.power(v0[i],2) * np.power(sin_theta[i],2) / (2*g)
    #Y[i] =(y[i]-y.min())/(y.max()-y.min())
    #Y[i] = y[i]-np.mean(y)*np.std(y)
    V0[i] =(v0[i]-v0.min())/(v0.max()-v0.min())
    Theta[i] =(theta[i]-theta.min())/(theta.max()-theta.min())
   # sin_2Theta[i] =(sin_2theta[i]-sin_2theta.min())/(sin_2theta.max()-sin_2theta.min()) 
    
for i in range(n):
    Y[i] =y[i]/y_max
   # th[i] =(y[i]-y.min())/(y.max()-y.min()) 

inp = np.hstack((V0,Theta))
inp = np.float32(inp)
Y = np.float32(Y)
#inp = np.hstack((V0,sin_2Theta))

inp_train, Y_train = inp[:90000], Y[:90000]    # first 160 data points
inp_test, Y_test = inp[90000:], Y[90000:]       # last 40 data points

  
 #---------------------------------------------------------------------add layer
def add_layer(inputs, in_size, out_size,n_layer,activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name= 'W')
            tf.summary.histogram(layer_name +  '/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs ,Weights ), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b,)
            tf.summary.histogram(layer_name + '/outputs',outputs)
        return outputs
#-----------------------------------------------------------------------    
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,shape=None,name='x_inp') 
    ys = tf.placeholder(tf.float32,shape=None,name='y_inp')
  
l1 = add_layer(xs, 2, 10,n_layer=1,activation_function=tf.sigmoid)
l2 = add_layer(l1, 10, 2,n_layer=2,activation_function=tf.sigmoid)
#l3 = add_layer(l2, 8, 6,activation_function=tf.sigmoid)
#l4 = add_layer(l3, 6, 4,activation_function=tf.sigmoid)
#l5 = add_layer(l4, 4, 2,activation_function=tf.sigmoid)

prediction = add_layer(l2, 2, 1,n_layer=3,activation_function=tf.sigmoid)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                                       reduction_indices=[1]))
  tf.summary.scalar('loss',loss)
  
with tf.name_scope('trian'):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
    
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)

sess.run(init)


for i in range(100):    
    
    sess.run(train_step, feed_dict={xs: inp_train, ys: Y_train})
    if i % 5 == 0:
        print('loss',sess.run(loss, feed_dict={xs: inp_train,  ys: Y_train}))
        result = sess.run(merged,feed_dict={xs: inp_train, ys: Y_train})
        writer.add_summary(result,i)
