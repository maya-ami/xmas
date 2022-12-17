import subprocess
from subprocess import Popen, PIPE
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import os
import pickle

import pandas as pd
import numpy as np
import json
import os
import re
import docx

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from PyPDF2 import PdfReader
from striprtf.striprtf import rtf_to_text

from pymystem3 import Mystem
import nltk
nltk.download('punkt')
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer

pp = Popen(['pwd'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
print(pp.communicate()[0].decode('utf-8'))

# pp = Popen(['ls /root/'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
# print(pp.communicate()[0].decode('utf-8'))

# os.environ['ANTIWORDHOME']="./"

MYDIR = os.path.dirname(__file__)

MODEL_FOLDER = 'models/'
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'doc', 'docx', 'pdf', 'rtf'}

data = pd.read_csv(os.path.join(MYDIR, 'DATA.csv'))

pipe = pickle.load(open(os.path.join(MYDIR,'models/model.pkl'), 'rb'))

legal_codes = {'ГК РФ': 'Гражданский кодекс РФ',
          'ТК РФ': 'Трудовой кодекс РФ',
         'НК РФ': 'Налоговый кодекс РФ',
         'КоАП РФ': 'Кодекс об административных правонарушениях РФ',
         'УК РФ': 'Уголовный кодекс РФ',
         'ГПК РФ': 'Гражданский процессуальный кодекс РФ',
         'АПК РФ':'Арбитражный процессуальный кодекс РФ',
         'ЖК РФ': 'Жилищный кодекс РФ',
         'СК РФ': 'Семейный кодекс РФ'}

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'


class FileTypeError(Exception):
    def __init__(self, msg='Неверный тип файла! Допустимые типы: doc, docx, pdf, rtf.'):
        super().__init__(msg)

def text_from_doc(filename):
    p = Popen(['/opt/venv/bin/antiword', '-f', '{}'.format(filename)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    text = output.decode('utf-8')
    return text

def text_from_docx(filename):
    doc = docx.Document(filename)
    fullText = []

    for para in doc.paragraphs:
        fullText.append(para.text)

    return '\n'.join(fullText)


def text_from_pdf(filename):
    reader = PdfReader(filename)

    num_pages = reader.getNumPages()
    text_pages = []

    for i in range(num_pages):
        page = reader.pages[i].extract_text()
        text_pages.append(page)

    return ''.join(text_pages)


def text_from_rtf(filename):
    with open(filename) as f:
        content = f.read()
        return rtf_to_text(content)


def extract_text(path):
    """
    Функция для выделения текста из документы. Допустимые типы файлов: doc, docx, pdf, rtf.

    Параметры:
        path: Путь к обрабатываемому файлу.
    """
    func_dict = {'doc': text_from_doc, 'docx': text_from_docx, 'pdf': text_from_pdf, 'rtf': text_from_rtf}

    try:
        filename = os.path.basename(path)
        extension = filename.split('.')[1]

        print('Обрабатывается файл ', path)

        if extension not in ['doc', 'docx', 'pdf', 'rtf']:
            raise FileTypeError
        else:
            text = func_dict[extension](path)

            return text
    except Exception as e:
        print(e)


def preprocess_no_lemm(line):
    """
    Функция предобработки текста:
    - очищает текст от цифр и лишних знаков препинания,
    - удаляет короткие слова (состоящие из 1 буквы),
    - удаяет стоп-слова,
    - проводит лемматизацию
    """
    char_regex = re.compile(r'[^а-яa-z\s]')
    line = char_regex.sub(' ', line.lower())

    short_words = re.compile(r'\b[а-яa-z]{1}\b')
    line = short_words.sub(' ', line.lower())

    return line.strip()


def phrases_by_class(doc_class):

    # все классы
    text = ''
    for i in data['clean_text'].values:
        text += i+' '

    all_w = nltk.tokenize.word_tokenize(text.lower())
    all_w_b = nltk.FreqDist(nltk.bigrams(w.lower() for w in all_w))
    all_w_t = nltk.FreqDist(nltk.trigrams(w.lower() for w in all_w))# if w not in russian_stopwords))

    allcl = []
    for k,v in dict(all_w_t.most_common(200)).items():
        allcl.append(' '.join(k))


    # конкретный класс
    text = ''
    for i in data[data['class']==doc_class]['clean_text'].values:
        text += i+' '

    all_w = nltk.tokenize.word_tokenize(text.lower())
    all_w_b = nltk.FreqDist(nltk.bigrams(w.lower() for w in all_w))
    all_w_t = nltk.FreqDist(nltk.trigrams(w.lower() for w in all_w))# if w not in russian_stopwords))

    cl = []
    for k,v in dict(all_w_t.most_common(200)).items():
        cl.append(' '.join(k))

    diff_rent = set(cl) - set(allcl)
    return diff_rent


def predict_with_keyphrases(doc):
    prediction, proba = pipe.predict([doc]), pipe.predict_proba([doc])
    proba_val = round(max(proba[0])*100, 1)
    print('Прогнозируемый тип договора: %s' %prediction[0])
    print('Прогнозная вероятность {}%'.format(proba_val))
    print('')
    print('В документе встречаются следующие формулировки:')
    print('')
    key_phrases = []
    for phrase in phrases_by_class(prediction[0]):
        if phrase in doc:
            print(phrase)
            key_phrases.append(phrase)
    print('**********************************************')
    print('')
    return prediction[0], proba_val, key_phrases

def check_for_legal_codes(doc):
    found_codes = []
    for k, v in legal_codes.items():
        if k.lower() in doc:
            found_codes.append(v)

    return found_codes


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
            file.save(os.path.join(MYDIR + "/" + app.config['UPLOAD_FOLDER'], filename))

            raw_text = extract_text(os.path.join(MYDIR + "/" + app.config['UPLOAD_FOLDER'], filename))
            clean_text = preprocess_no_lemm(raw_text)
            prediction, predict_proba, key_phrases = predict_with_keyphrases(clean_text)
            found_codes = check_for_legal_codes(clean_text)

        else:
            flash('Убедитесь, что отправляете файл формата doc, docx или pdf.')
        return render_template('result.html', prediction=prediction,
                            predict_proba=predict_proba, key_phrases=', '.join(key_phrases),
                            found_codes=', '.join(found_codes))

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
    app.run('0.0.0.0', debug=False)
