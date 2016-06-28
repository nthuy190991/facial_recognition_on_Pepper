# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:43:38 2016

@author: thnguyen
"""

import numpy as np
import os, sys
import time
from threading import Thread
from flask import Flask, request, render_template, send_from_directory
import requests
import random

_username   = 'thanhhuynguyenorange'
_password   = 'GGQN0871abc'
_url_github = 'https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford'

def get_list_images():
    res_list_img = requests.get(_url_github, auth=(_username, _password))
    res_list_img = eval(res_list_img.content)

    list_images = []
    for img in res_list_img:
        list_images.append(img['name'])

    # image_paths = [imgPathRemote+f for f in list_images if f.startswith(name)]
    return list_images

def run_program(clientId):
    global global_vars
    global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    # global_var['name'] = 'huy' # to be replaced by recognition

    if (global_var['flag_local_remote']):
        list_images = get_list_images()
        list_names = [f for f in list_images if f.endswith('.0.jpg')]
        print list_names

        ind = random.randint(0,len(list_names)-1)
        print ind
        global_var['name'] = list_names[ind][0:len(list_names[ind])-6]

    else:
        list_names = [f for f in os.listdir(imgPathLocal) if f.endswith('.0.png')]

        ind = random.randint(0,len(list_names)-1)
        global_var['name'] = list_names[ind][0:len(list_names[ind])-6]

    show_photos(clientId)


def show_photos(clientId):
    global global_vars
    global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    if (global_var['flag_local_remote']):
        list_images = get_list_images()
        image_paths = [imgPathRemote+f for f in list_images if f.startswith(global_var['name'])]
        for img_path in image_paths:
            global_var['todo'] = "R https://github.com/nthuy190991/facial_recognition_on_Bluemix/blob/master/" + img_path + "?raw=true"
            while (global_var['todo'] != ""):
                pass
    else:
        image_paths = [os.path.join(imgPathLocal, f) for f in os.listdir(imgPathLocal) if f.startswith(global_var['name'])]
        for img_path in image_paths:
            img_path = img_path.replace('\\', '/')
            global_var['todo'] = "L " + img_path
            while (global_var['todo'] != ""):
                pass

"""
Initialisation for global variables used by clientId
"""
def global_var_init(clientId):

    global global_vars
  
    todo = ''
    name = ''
    
    flag_local_remote = False # True: remote
                             # False: local

    global_vars.append(dict([   ('clientId', str(clientId)),
                                ('name', name),
                                ('flag_local_remote', flag_local_remote),
                                ('todo', todo) 
                            ]))

"""
==============================================================================
Flask Initialization
==============================================================================
"""
def flask_init():
    global app

    app  = Flask(__name__, static_url_path='')

    @app.route('/')
    def render_hmtl():
        return render_template('index.html')

    @app.route('/css/<path:filename>')
    def sendCSS(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'templates', 'css'), filename)

    @app.route('/js/<path:filename>')
    def sendJavascript(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'templates', 'js'), filename)

    @app.route('/img/<path:filename>')
    def sendImages(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'templates', 'img'), filename)

    @app.route('/face_database/<path:filename>')
    def sendPhotosFromDatabase(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'face_database'), filename)

    @app.route('/start', methods=['POST'])
    def onStart():
        clientId = request.args.get('id')
        global_var_init(clientId)
        
        thread_program = Thread(target = run_program, args= (clientId,), name = 'thread_prog_'+clientId)
        thread_program.start()        

        return "", 200

    @app.route('/wait', methods=['POST'])
    def waitForServerInput():
        clientId = request.args.get('id')
        time.sleep(0.1)

        global global_vars
        global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()
        temp        = global_var['todo']
        global_var['todo'] = ""

        return temp, 200

"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""

imgPathRemote = "face_database_for_oxford/"
imgPathLocal  = "face_database/"

global_vars = []

flask_init()

port = int(os.getenv('PORT', '9099'))
app.run(host = '0.0.0.0', port = port, threaded = True)
