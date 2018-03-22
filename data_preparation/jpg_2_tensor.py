import tensorflow as tf
import numpy as np
from PIL import Image
import random

#jpg_2_tensor.py

#画像のファイルを配列にする関数

def jpg_2_tensor():
　#正解ラベル
    right_rabels_latte=[]
    right_rabels_other=[]
    #画像数
    latte_gazou=901
    other_gazou=1101
    gokei=latte_gazou+other_gazou

    #戻り値として配列が入る
    latte_r=[]
    other_r=[]

    for a in range(latte_gazou):
        latte_r.append(0)
    for a in range(other_gazou):
        other_r.append(0)


    for a in range(latte_gazou):
        right_rabels_latte.append([0,1])
        b=str(a+1)
        #画像読み込み
        latte_r[a]=Image.open("./data/latte/l("+b+").jpg")
        a=int(a)
        # オリジナル画像の幅と高さを取得
        width, height = latte_r[a].size

        img_pixels = []
        for y in range(height):
            for x in range(width):
                # getpixel((x,y))で左からx番目,上からy番目のピクセルの色を取得し、img_pixelsに追加する
                img_pixels.append(latte_r[a].getpixel((x, y)))
        # numpyのarrayに変換する
        latte_r[a]=np.array(img_pixels)
        #(28,28),RGBの配列にする
        latte_r[a]=np.reshape(latte_r[a], (28, 28, 3))

    for b in range(other_gazou):
        right_rabels_other.append([1,0])
        c=str(b+1)
        #画像を読み込む
        other_r[b]=Image.open("./data/other/o("+c+").jpg")
        b=int(b)
    #オリジナル画像の幅と高さを取得
        width, height = other_r[b].size
        img_pixels = []
        for y in range(height):
            for x in range(width):
        # getpixel((x,y))で左からx番目,上からy番目のピクセルの色を取得し、img_pixelsに追加する
                img_pixels.append(other_r[b].getpixel((x,y)))
        if(not(isinstance(img_pixels[0],int))):

            if(len(img_pixels[0])==3):

                # numpyのarrayに変換する
                other_r[b] = np.array(img_pixels)
                #(28,28),RGBの配列にする
                other_r[b]=np.reshape(other_r[b],(28,28,3))
        else:
            for a in range(784):
                img_pixels[a]=[img_pixels[a],1,1]
            #numpyに変換する
            other_r[b] = np.array(img_pixels)
            #(28,28),RGBの配列にする
            other_r[b]=np.reshape(other_r[b],(28,28,3))
    #それぞれlatte,otherの全画像が入った配列のリストを返す
    return latte_r,other_r