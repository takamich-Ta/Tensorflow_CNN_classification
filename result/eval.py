import tensorflow as tf
from PIL import Image
import numpy as np


#eval.py


#�w�K�ς݃p�����[�^�[��p���āA�摜���N���X���ނ���֐�

def evaluation(img_path):

    #�摜��ǂݍ���ŁA�z��ɒ���

    image_r=[0]

    image_r[0] = Image.open(img_path)
    #(28,28)�ɒ���
    image_r[0]=image_r[0].resize((28,28))

    # �I���W�i���摜�̕��ƍ������擾
    width, height = image_r[0].size

    img_pixels = []
    for y in range(height):
        for x in range(width):
            # getpixel((x,y))�ō�����x�Ԗ�,�ォ��y�Ԗڂ̃s�N�Z���̐F���擾���Aimg_pixels�ɒǉ�����
            img_pixels.append(image_r[0].getpixel((x, y)))

    #RGB���ǂ������肵�āA���ꂼ�ꏈ��
    if (not (isinstance(img_pixels[0], int))):

        if (len(img_pixels[0]) == 3):
            # numpy��array�ɕϊ�����
            image_r[0] = np.array(img_pixels)
            #(28,28),RGB�ɕό`
            image_r[0] = np.reshape(image_r[0], (1,28, 28, 3))
    else:
        for a in range(784):
            #�O���[�X�P�[���͖����RGB�ɂ���
            img_pixels[a] = [img_pixels[a], 1, 1]
        image_r[0] = np.array(img_pixels)
        image_r[0] = np.reshape(image_r[0], (1,28, 28, 3))

    #�z����N���X���ނ���
    #�d�ݍ��
    def weight_variable(shape,name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial,name=name)
    #�o�C�A�X���
    def bias_variable(shape,name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial,name=name)
    #��ݍ��ݍ��
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
     #�v�[�����O���
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    #���͗p
    x=tf.placeholder(tf.float32)

    #weight1,bias1,5*5�̃t�B���^�[,RGB3,�o��32

    W_conv1 = weight_variable([5, 5, 3, 32],"W_conv1")
    b_conv1 = bias_variable([32],"b_conv1")


    #��ݍ��݉��Z1,�v�[�����O1

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    #��ڂ�weight,bias

    W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
    b_conv2 = bias_variable([64],"b_conv2")

    #��ݍ���2,�v�[�����O2

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #weight3,bias3

    W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
    b_fc1 = bias_variable([1024],"b_fc1")

    # �S�����w�@�ꎟ���ɒ���

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    #�h���b�v�A�E�g

    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    #weight4,bias4

    W_fc2 = weight_variable([1024, 2],"W_fc2")
    b_fc2 = bias_variable([2],"b_fc2")

    #�o��

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #latte�̐������x��

    right_rabel=[0,1]

    #�w�K�ς݃f�[�^�ǂݍ���

    ckpt_path="./model4.ckpt"
    sess=tf.InteractiveSession()
    saver=tf.train.Saver()
    saver.restore(sess,ckpt_path)

    #������m�F

    right=tf.constant(right_rabel)
    #�������x���ƈ�v���Ă��邩���ׂ�
    correct_prediction=tf.equal(tf.argmax(right,0),tf.argmax(y_conv,0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    decision=sess.run(accuracy,feed_dict={x:image_r[0],keep_prob:1.0})
    result="failed"
    if(decision==1.0):
        result="latte"
    if(decision==0.0):
        result="other"

    return result