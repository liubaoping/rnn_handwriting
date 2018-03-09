# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:41:39 2018

@author: l
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#输入图片是28*28
n_inputs=28 #输入一行，一行有28个数据
max_time=28 #一共28行
lstm_size=100#隐层单元
n_classes=10#十个分类
batch_size=50#每批次50个样本
n_batch=mnist.train.num_examples//batch_size

#none表示第一个维度可以为任意长度
x=tf.placeholder(tf.float32,[None,784])#28*28
y=tf.placeholder(tf.float32,[None,10])

#初始化权值
weights=tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
biases=tf.Variable(tf.constant(0.1,shape=[n_classes]))

def RNN(X,weights,biases):
    #inputs=[batch_size,max_time,n_inputs]
    inputs=tf.reshape(X,[-1,max_time,n_inputs])
    #定义LSTM基本cell
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(lstm_size)
    #final_state[0]是cell_state
    #final_state[1]是hidden_state
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results
    
#计算rnn的返回结果
prediction=RNN(x,weights,biases)

#损失函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))

#使用AdamOptimize进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 
#初始化变量
init=tf.global_variables_initializer()

#结果存放在布尔列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回最大值所在的位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range (n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        #train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1})        
        print("Iter:"+str(epoch)+"  Test_Accuracy:"+str(test_acc))

    