from flask import Flask, render_template, request
from werkzeug import secure_filename
import eval

#app.py

app = Flask(__name__)

#�z�[�����

#�z�[�����
@app.route('/')
def upload_file():
    #�f�B���N�g��templates��upload.html���o��
    return render_template('upload.html')

#�摜�A�b�v���[�h���
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        #�����̃f�B���N�g����a.jpg�Ƃ��ĕۑ�����
        f.save(secure_filename("a.jpg"))
        #�֐��ɂ�锻���߂�l
        return eval.evaluation("a.jpg")

if __name__ == '__main__':
    app.run(debug=True)