from flask import Flask, render_template, request
from werkzeug import secure_filename
import eval

#app.py

app = Flask(__name__)

#ホーム画面

#ホーム画面
@app.route('/')
def upload_file():
    #ディレクトリtemplatesのupload.htmlを出力
    return render_template('upload.html')

#画像アップロード画面
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        #自分のディレクトリにa.jpgとして保存する
        f.save(secure_filename("a.jpg"))
        #関数による判定を戻り値
        return eval.evaluation("a.jpg")

if __name__ == '__main__':
    app.run(debug=True)