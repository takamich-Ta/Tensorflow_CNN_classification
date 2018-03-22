import tensorflow as tf
from PIL import Image
import numpy as np


#eval.py


#学習済みパラメーターを用いて、画像をクラス分類する関数

def evaluation(img_path):

    #画像を読み込んで、配列に直す

    image_r=[0]

    image_r[0] = Image.open(img_path)
    #(28,28)に直す
    image_r[0]=image_r[0].resize((28,28))

    # オリジナル画像の幅と高さを取得
    width, height = image_r[0].size

    img_pixels = []
    for y in range(height):
        for x in range(width):
            # getpixel((x,y))で左からx番目,上からy番目のピクセルの色を取得し、img_pixelsに追加する
            img_pixels.append(image_r[0].getpixel((x, y)))

    #RGBかどうか判定して、それぞれ処理
    if (not (isinstance(img_pixels[0], int))):

        if (len(img_pixels[0]) == 3):
            # numpyのarrayに変換する
            image_r[0] = np.array(img_pixels)
            #(28,28),RGBに変形
            image_r[0] = np.reshape(image_r[0], (1,28, 28, 3))
    else:
        for a in range(784):
            #グレースケールは無理矢理RGBにする
            img_pixels[a] = [img_pixels[a], 1, 1]
        image_r[0] = np.array(img_pixels)
        image_r[0] = np.reshape(image_r[0], (1,28, 28, 3))

    #配列をクラス分類する
    #重み作る
    def weight_variable(shape,name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial,name=name)
    #バイアス作る
    def bias_variable(shape,name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial,name=name)
    #畳み込み作る
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
     #プーリング作る
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    #入力用
    x=tf.placeholder(tf.float32)

    #weight1,bias1,5*5のフィルター,RGB3,出力32

    W_conv1 = weight_variable([5, 5, 3, 32],"W_conv1")
    b_conv1 = bias_variable([32],"b_conv1")


    #畳み込み演算1,プーリング1

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    #二つ目のweight,bias

    W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
    b_conv2 = bias_variable([64],"b_conv2")

    #畳み込み2,プーリング2

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #weight3,bias3

    W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
    b_fc1 = bias_variable([1024],"b_fc1")

    # 全結合層　一次元に直す

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    #ドロップアウト

    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    #weight4,bias4

    W_fc2 = weight_variable([1024, 2],"W_fc2")
    b_fc2 = bias_variable([2],"b_fc2")

    #出力

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #latteの正解ラベル

    right_rabel=[0,1]

    #学習済みデータ読み込み

    ckpt_path="./model4.ckpt"
    sess=tf.InteractiveSession()
    saver=tf.train.Saver()
    saver.restore(sess,ckpt_path)

    #判定を確認

    right=tf.constant(right_rabel)
    #正解ラベルと一致しているか調べる
    correct_prediction=tf.equal(tf.argmax(right,0),tf.argmax(y_conv,0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    decision=sess.run(accuracy,feed_dict={x:image_r[0],keep_prob:1.0})
    result="failed"
    if(decision==1.0):
        result="latte"
    if(decision==0.0):
        result="other"

    return result