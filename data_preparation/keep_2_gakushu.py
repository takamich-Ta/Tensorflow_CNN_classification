from PIL import Image
import os

#keep_2_gakushu.py

#�摜�̖���
#data_keep_other_gazou=1101
#data_keep_latte_gazou=901

#data_keep�̉摜��(28,28)�Ƀ��T�C�Y���āAdata�Ɉړ�

def keep_2_other(data_keep_other_gazou,name):

    for a in range(data_keep_other_gazou):
        b=str(a+1)
        #�p�X���w�肵�ĉ摜�ǂݍ���
        image=Image.open("./data_keep/other/"+name+" ("+b+").jpg")
        #�摜��(28,28)�ɂ���
        image=image.resize((28,28))
        #�ړ���ɂ��łɉ摜����������A������������ĕۑ�
        if(os.path.isfile("./data/other/"+name+"("+b+").jpg")):
            os.remove("./data/other/"+name+"("+b+").jpg")
        #�摜�ۑ�
        image.save("./data/other/"+name+"("+b+").jpg")

def keep_2_latte(data_keep_latte_gazou,name):
    for a in range(data_keep_latte_gazou):
        b=str(a+1)
        #�摜��ǂݍ���
        image=Image.open("./data_keep/latte"+name+ "("+b+").jpg")
        #�摜��(28,28)�ɂ���
        image=image.resize((28,28))
        #�ړ���ɉ摜����������A������������ĕۑ�
        if(os.path.isfile("./data/latte/"+name+"("+b+").jpg")):
            os.remove("./data/latte/"+name+"("+b+").jpg")
        #�摜��ۑ�
        image.save("./data/latte/"+name+"("+b+").jpg")