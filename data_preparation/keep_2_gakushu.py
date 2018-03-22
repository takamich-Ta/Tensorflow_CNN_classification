from PIL import Image
import os

#keep_2_gakushu.py

#画像の枚数
#data_keep_other_gazou=1101
#data_keep_latte_gazou=901

#data_keepの画像を(28,28)にリサイズして、dataに移動

def keep_2_other(data_keep_other_gazou,name):

    for a in range(data_keep_other_gazou):
        b=str(a+1)
        #パスを指定して画像読み込み
        image=Image.open("./data_keep/other/"+name+" ("+b+").jpg")
        #画像を(28,28)にする
        image=image.resize((28,28))
        #移動先にすでに画像があったら、それを消去して保存
        if(os.path.isfile("./data/other/"+name+"("+b+").jpg")):
            os.remove("./data/other/"+name+"("+b+").jpg")
        #画像保存
        image.save("./data/other/"+name+"("+b+").jpg")

def keep_2_latte(data_keep_latte_gazou,name):
    for a in range(data_keep_latte_gazou):
        b=str(a+1)
        #画像を読み込み
        image=Image.open("./data_keep/latte"+name+ "("+b+").jpg")
        #画像を(28,28)にする
        image=image.resize((28,28))
        #移動先に画像があったら、それを消去して保存
        if(os.path.isfile("./data/latte/"+name+"("+b+").jpg")):
            os.remove("./data/latte/"+name+"("+b+").jpg")
        #画像を保存
        image.save("./data/latte/"+name+"("+b+").jpg")