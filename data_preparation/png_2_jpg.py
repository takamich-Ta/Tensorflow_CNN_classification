from PIL import Image
import os

#png_2_jpg.py

#png�̉摜�̖���
png_gazou=82

#png��jpg�ɕϊ�����

def png_2_jpg(png_gazou,name):

    for a in range(png_gazou):
        b=str(a+1)
        #�p�X���w�肵�ĉ摜�ǂݍ���
        img=Image.open("./data_keep/other/"+name+" ("+b+").png")
        width,height=img.size
        canvas=Image.new("RGB",(width,height),(255,255,255))
        canvas.paste(img,(0,0))
        #jpg�Ƃ��ĕۑ�
        canvas.save("./data_keep/other/"+name+"("+b+").jpg","JPEG",quality=100,optimizer=True)
 