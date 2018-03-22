import tensorflow as tf
import numpy as np
from PIL import Image
import random
#jpgを配列に直す関数
import jpg_2_tensor
#配列をテストデータ、学習用データに保存する関数
import tensor_2_all


#new_20171111_keep_20171122.py


#テストデータ、テストラベル、学習用データ、学習用ラベル用意
test_r,test_rabels,train_r,train_rabels=tensor_2_all.tensor_2_all()

#入力用プレースホルダー
x=tf.placeholder(tf.float32)

#重み作る
def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)
#バイアス作る
def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)
畳み込み演算
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#プーリング演算
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



#一つ目のweight,bias,5*5のフィルター,RGB3,出力32

W_conv1 = weight_variable([5, 5, 3, 32],"W_conv1")
b_conv1 = bias_variable([32],"b_conv1")


#畳み込み1、プーリング1

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#二つ目のweight,bias

W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
b_conv2 = bias_variable([64],"b_conv2")

#畳み込み2、プーリング2

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#三つめのweight,bias

W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
b_fc1 = bias_variable([1024],"b_fc1")

#全結合層　一次元に直す

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#ドロップアウト

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#四つ目のweight,bias

W_fc2 = weight_variable([1024, 2],"W_fc2")
b_fc2 = bias_variable([2],"b_fc2")

#出力

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#正解ラベル

right_rabel=tf.placeholder(tf.float32)

#損失関数

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=right_rabel, logits=y_conv))

#交差エントロピーを減らす
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#学習する
print()
tr=tf.placeholder(tf.float32)

#正解ラベルとの一致を調べる
correct_prediction=tf.equal(tf.argmax(tr,1),tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#バッチ入れる箱
batch=[]
batch_rabels=[]

sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(int(1000)):
    #バッチを作る
    for a in range(20):
        b=random.randint(0,1972-1)
        batch.append(train_r[b])
        batch_rabels.append(train_rabels[b])
    #学習実行
    sess.run(train_step,feed_dict={x:batch,right_rabel:batch_rabels,keep_prob:0.5})
    if(i%100==0):
        #出力を出力
        print(sess.run(y_conv,feed_dict={x:test_r,right_rabel:test_rabels,keep_prob:1.0}))

    #テストデータで確認する

    if(i%10==0):
        print([i, int(1000)-1])
        print(sess.run(accuracy,feed_dict={x:test_r,tr:test_rabels,keep_prob:1.0}))

    #最終的な精度

    if(i==int((1000)-1)):
        print(sess.run(accuracy, feed_dict={x: test_r, tr: test_rabels, keep_prob: 1.0}))

#学習済みデータを保存

saver=tf.train.Saver()
saver.save(sess, "./model4.ckpt")
 