import tensorflow as tf
import numpy as np
from PIL import Image
import random
import jpg_2_tensor

#tensor_2_all.py

#(28,28,3)���������z��,latte_r,other_r���w�K�p�f�[�^�A�e�X�g�p�f�[�^�ɕ�����

def tensor_2_all():
    #901,1101,30
    test=30
    #�e�X�g30�����Alatte��15
    latte_test=int(test/2)
    #other��15
    other_test=int(test/2)
    latte_train=901-latte_test
    other_train=1101-other_test

    latte_r, other_r=jpg_2_tensor.jpg_2_tensor()
    #�e�X�g�p�f�[�^�A�e�X�g�p�f�[�^���x���A�w�K�p�f�[�^�A�w�K�p�f�[�^���x��
    test_r=[]
    test_rabels=[]
    train_r=[]
    train_rabels=[]

    #���ꂼ��ɕۑ��A�f�[�^�A���x��
    for a in range(latte_test):
        test_r.append(latte_r[a])
        test_rabels.append([0,1])
    for a in range(other_test):
        test_r.append(other_r[a])
        test_rabels.append([1,0])
    for a in range(latte_train):
        train_r.append(latte_r[a+latte_test])
        train_rabels.append([0,1])
    for a in range(other_train):
        train_r.append(other_r[a+other_test])
        train_rabels.append([1,0])
    #�f�[�^�A���x����Ԃ��A�e�X�g�p�A�w�K�p
    return test_r,test_rabels,train_r,train_rabels