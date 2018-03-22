from PIL import Image
import os

#png_2_jpg.py

#png‚Ì‰æ‘œ‚Ì–‡”
png_gazou=82

#png‚ğjpg‚É•ÏŠ·‚·‚é

def png_2_jpg(png_gazou,name):

    for a in range(png_gazou):
        b=str(a+1)
        #ƒpƒX‚ğw’è‚µ‚Ä‰æ‘œ“Ç‚İ‚İ
        img=Image.open("./data_keep/other/"+name+" ("+b+").png")
        width,height=img.size
        canvas=Image.new("RGB",(width,height),(255,255,255))
        canvas.paste(img,(0,0))
        #jpg‚Æ‚µ‚Ä•Û‘¶
        canvas.save("./data_keep/other/"+name+"("+b+").jpg","JPEG",quality=100,optimizer=True)
 