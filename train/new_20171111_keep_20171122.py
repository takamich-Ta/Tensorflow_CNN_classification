import tensorflow as tf
import numpy as np
from PIL import Image
import random
#jpg��z��ɒ����֐�
import jpg_2_tensor
#�z����e�X�g�f�[�^�A�w�K�p�f�[�^�ɕۑ�����֐�
import tensor_2_all


#new_20171111_keep_20171122.py


#�e�X�g�f�[�^�A�e�X�g���x���A�w�K�p�f�[�^�A�w�K�p���x���p��
test_r,test_rabels,train_r,train_rabels=tensor_2_all.tensor_2_all()

#���͗p�v���[�X�z���_�[
x=tf.placeholder(tf.float32)

#�d�ݍ��
def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)
#�o�C�A�X���
def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)
��ݍ��݉��Z
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#�v�[�����O���Z
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



#��ڂ�weight,bias,5*5�̃t�B���^�[,RGB3,�o��32

W_conv1 = weight_variable([5, 5, 3, 32],"W_conv1")
b_conv1 = bias_variable([32],"b_conv1")


#��ݍ���1�A�v�[�����O1

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#��ڂ�weight,bias

W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
b_conv2 = bias_variable([64],"b_conv2")

#��ݍ���2�A�v�[�����O2

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#�O�߂�weight,bias

W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
b_fc1 = bias_variable([1024],"b_fc1")

#�S�����w�@�ꎟ���ɒ���

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#�h���b�v�A�E�g

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#�l�ڂ�weight,bias

W_fc2 = weight_variable([1024, 2],"W_fc2")
b_fc2 = bias_variable([2],"b_fc2")

#�o��

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#�������x��

right_rabel=tf.placeholder(tf.float32)

#�����֐�

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=right_rabel, logits=y_conv))

#�����G���g���s�[�����炷
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#�w�K����
print()
tr=tf.placeholder(tf.float32)

#�������x���Ƃ̈�v�𒲂ׂ�
correct_prediction=tf.equal(tf.argmax(tr,1),tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#�o�b�`����锠
batch=[]
batch_rabels=[]

sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(int(1000)):
    #�o�b�`�����
    for a in range(20):
        b=random.randint(0,1972-1)
        batch.append(train_r[b])
        batch_rabels.append(train_rabels[b])
    #�w�K���s
    sess.run(train_step,feed_dict={x:batch,right_rabel:batch_rabels,keep_prob:0.5})
    if(i%100==0):
        #�o�͂��o��
        print(sess.run(y_conv,feed_dict={x:test_r,right_rabel:test_rabels,keep_prob:1.0}))

    #�e�X�g�f�[�^�Ŋm�F����

    if(i%10==0):
        print([i, int(1000)-1])
        print(sess.run(accuracy,feed_dict={x:test_r,tr:test_rabels,keep_prob:1.0}))

    #�ŏI�I�Ȑ��x

    if(i==int((1000)-1)):
        print(sess.run(accuracy, feed_dict={x: test_r, tr: test_rabels, keep_prob: 1.0}))

#�w�K�ς݃f�[�^��ۑ�

saver=tf.train.Saver()
saver.save(sess, "./model4.ckpt")
 