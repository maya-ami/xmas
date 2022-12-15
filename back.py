import subprocess
from subprocess import PIPE, Popen
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = '/Users/mayabikmetova/xmas'
ALLOWED_EXTENSIONS = {'doc', 'docx', 'pdf'}

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def start():
    return render_template('start.html')

@app.route('/result',methods=['GET','POST'])
def result():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            subprocess.call(["vosk-transcriber -l ru -i {} -o {}.txt".format(filename, filename)], shell=True)
            text = []
            with open('{}.txt'.format(filename), 'r') as f:
                for line in f:
                    text.append(line)
        else:
            flash('Убедитесь, что отправляете файл формата wav, mp3 или mp4.')
        # # Отправляем wav файл на ASR сервис
        # text = requests.get("http://0.0.0.0:5000/recognize_wav")
        return render_template('result.html', text=text)

@app.route('/download_results')
def download_results():
    return send_file('Final.csv',
                     mimetype='text/csv',
                     attachment_filename='Final.csv',
                     as_attachment=True)


@app.errorhandler(500)
def internalServerError(e):
    return render_template('500.html'), 500

@app.errorhandler(404)
def internalServerError(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
