from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pickle import dump, load
from keras.models import Model, load_model
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
import os
from pathlib import Path
from googletrans import Translator
from gtts import gTTS
from django.core.files.storage import FileSystemStorage
translator = Translator(service_urls=['translate.googleapis.com','translate.google.com','translate.google.co.kr'])
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
model=load_model(os.path.join(BASE_DIR,"model_9_1.h5"))
tokenizer = load(open(os.path.join(BASE_DIR,"tokenizer_1.p"),"rb"))
text=""
max_length = 32
xception_model = Xception(include_top=False, pooling="avg")

def index(request):
    return render(request, "main_page/index.html")

def predict(request):
    global text
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            os.remove(os.path.join(BASE_DIR,"static/main_page/img.jpg"))
            filename = fs.save(os.path.join(BASE_DIR,"static/main_page/img.jpg"), myfile)
            photo = extract_features(filename, xception_model)
            description = generate_desc(model, tokenizer, photo, max_length)
            words = description.split()
            red="red"
            filtered_sentence = [w for w in words if not w in red]
            textl=filtered_sentence[1:-1]
            text = ' '.join([str(elem) for elem in textl])
            print(text)
            result = translator.translate(text, src='en', dest='ne')
            text=result.text
            tts = gTTS(text=result.text,lang='ne',slow=False)
            tts.save(os.path.join(BASE_DIR,'static/main_page/sound.mp3'))
            return render(request, "main_page/predict.html",{"text":text})


    except Exception as e:
        return HttpResponse("<h1> Please upload image for auto caption</h1>") 

def extract_features(filename, model):
        try:
            image = load_img(filename, target_size=(299, 299))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image)
            return feature
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
             return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text
