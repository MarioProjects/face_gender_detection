from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import json

# Ignore warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

""" --- IMPORT LIBRARIES --- """

import numpy as np
import pickle
import pathlib
import time

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
import torch.nn.functional as F

import cv2

from models import models_interface
import face_recognition

CAT2CLASS = {0:"Male", 1:"Female"}

""" -- MODEL LOAD -- """

# We establish a seed for the replication of the experiments correctly
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

model_type, model_cfg, optimizador, batch_size, flat_size, block_type, last_pool_size = "MobileNetv2", "MobileNetSmallv0", "SGD", 16, 512, "", 5
num_classes = 2
dropout, ruido, input_channels = 0.0, 0.0, 3
growth_rate, in_features = 0, 0
out_type, block_type = "relu", ""
slack_channel = "log_ai_work"

if type(model_cfg)==list or type(model_cfg)==tuple:
    model_cfg_txt = '_'.join(model_cfg)
else: model_cfg_txt = model_cfg

print("{} - {} using {} - LFW - Color!)".format(model_type, str(model_cfg_txt), optimizador))
states_path = "models/CE_Simple_checkpoint_state_mobilenetsmall_color.pt"
MODEL = models_interface.load_model(model_type, states_path=states_path, model_config=model_cfg, dropout=dropout, ruido=ruido, input_channels=input_channels, growth_rate=growth_rate, flat_size=flat_size, in_features=in_features, out_type=out_type, block_type=block_type, out_features=num_classes, last_pool_size=last_pool_size)
MODEL = MODEL.cpu()
MODEL.eval()

SOFTMAX = nn.Softmax()

# Define a flask app
app = Flask(__name__)

def apply_img_albumentation(aug, image):
    image = aug(image=image)['image']
    return image

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictModel', methods=['GET', 'POST'])
def predictModel():
    if request.method == 'POST':
        # I take the path where the image has been uploaded and saved locally
        file_path = get_file_path_and_save(request)

        # I load the image and treat it normally (I am on Python!)
        img = face_recognition.load_image_file(file_path)

        # Remove file to no store trash
        os.remove(file_path)

        # We go to the detection of the faces to extract them with face_recognition
        # https://github.com/ageitgey/face_recognition
        print("Ha llegado una imagen nueva! Procedemos a detectar los rostros...")
        face_locations = face_recognition.face_locations(img)
        print("{} rostros detectados!".format(len(face_locations)))

        # If we have not found faces, we will finish and communicate
        if len(face_locations)==0: return json.dumps({"meta_info":"No se han encontrado rostros"})

        # If faces have been found, we process them one by one
        info = {}
        info["faces"]={}
        male_count, female_count = 0, 0
        for face_location in face_locations:

            # Print the location of each face in this image
            top, right, bottom, left = face_location

            # We take the face and transform it and cast into a torch format
            face = img[top:bottom, left:right]
            face = cv2.resize(face, (80, 80)) 
            face = torch.from_numpy(face.transpose(2,0,1))
            face = face.unsqueeze(0).type('torch.FloatTensor')

            # We make the prediction of the current face
            with torch.no_grad():
                prediction = MODEL(face.cpu())
            preds_classes = torch.argmax(prediction, dim=1)
            confianza = SOFTMAX(prediction)

            # We take out the information from the prediction
            genero = CAT2CLASS[preds_classes[0].item()]
            confianza = confianza[0][preds_classes[0].item()].item()*100

            if genero.lower()=="female": female_count+=1
            else: male_count+=1

            # I store in an object the necessary information that we will return to the browser
            info["faces"].update({"face"+str(male_count+female_count): {"x":left, "y":top, "width":bottom-top, "height":right-left, "gender":genero}})

            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {} - {} con confianza {:.2f}".format(top, left, bottom, right, genero, confianza))

        info["meta_info"] = "Encontrado rostros -> Masculinos " + str(male_count) + " y Femeninos " + str(female_count)

        # We return to the browser what we find as a json object
        return json.dumps(info)
    return None


if __name__ == '__main__':
    # Serve the app with gevent
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    print("\n########################################")
    print('--- Running on port {} ---'.format(port))
    print("########################################\n")
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()