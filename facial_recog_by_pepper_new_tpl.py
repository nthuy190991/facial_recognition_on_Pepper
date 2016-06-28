# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:43:38 2016

@author: thnguyen
"""

import numpy as np
import os, sys
import cv2
import time
from read_xls import read_xls
#from edit_xls import edit_xls
import xlrd
from threading import Thread
from flask import Flask, request, render_template, send_from_directory
import operator
from binascii import b2a_hex
from watson_developer_cloud import NaturalLanguageClassifierV1
import face_api
import emotion_api
import qi # Aldebaran Python SDK

"""
Replace French accents in texts
"""
def replace_accents(text):
    chars_origine = ['Ê','à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é',
                     'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö',
                     'ù', 'ú', 'û', 'ü']
    chars_replace = ['\xC3','\xE0', '\xE1', '\xE2', '\xE3', '\xE4', '\xE5',
                      '\xE6', '\xE7', '\xE8', '\xE9', '\xEA', '\xEB', '\xEC',
                      '\xED', '\xEE', '\xEF', '\xF2', '\xF3', '\xF4', '\xF5',
                      '\xF6', '\xF9', '\xFA', '\xFB', '\xFC']
    text2 = str_replace_chars(text, chars_origine, chars_replace)
    return text2

def replace_accents2(text):
    chars_origine = ['Ê', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é',
                     'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö',
                     'ù', 'ú', 'û', 'ü']
    chars_replace = ['E', 'a', 'a', 'a', 'a', 'a', 'a', 'ae', 'c', 'e', 'e',
                     'e', 'e', 'i', 'i', 'i', 'i', 'o', 'o', 'o', 'o', 'o',
                     'u', 'u', 'u', 'u']
    text2 = str_replace_chars(text, chars_origine, chars_replace)
    return text2

"""
Replace characters in a string
"""
def str_replace_chars(text, chars_origine, chars_replace):
    for i in range(len(chars_origine)):
        text2 = text.replace(chars_origine[i], chars_replace[i])
        text  = text2
    return text2

"""
==============================================================================
Face and Emotion API
==============================================================================
"""
def retrieve_face_emotion_att(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    chrome_server2client(clientId, 'START')
    app_pepper.pepper_tts(clientId, 'Veuillez patienter pendant quelques secondes...')

    # Face API
    faceResult = face_api.faceDetect(None, 'output.png', None)

    # Emotion API
    emoResult = emotion_api.recognizeEmotion(None, 'output.png', None)

    # Results
    print 'Found {} '.format(len(faceResult)) + ('faces' if len(faceResult)!=1 else 'face')
    nb_faces = len(faceResult)
    tb_face_rect = [{} for ind in range(nb_faces)]
    tb_age       = ['' for ind in range(nb_faces)]
    tb_gender    = ['' for ind in range(nb_faces)]
    tb_glasses   = ['' for ind in range(nb_faces)]
    tb_emo       = ['' for ind in range(len(emoResult))]

    if (len(faceResult)>0 and len(emoResult)>0):
        ind = 0
        for currFace in faceResult:
            faceRectangle       = currFace['faceRectangle']
            faceAttributes      = currFace['faceAttributes']

            tb_face_rect[ind]   = faceRectangle
            tb_age[ind]         = str(faceAttributes['age'])
            tb_gender[ind]      = faceAttributes['gender']
            tb_glasses[ind]     = faceAttributes['glasses']
            ind += 1

        ind = 0
        for currFace in emoResult:
            tb_emo[ind] = max(currFace['scores'].iteritems(), key=operator.itemgetter(1))[0]
            ind += 1

        faceWidth  = np.zeros(shape=(nb_faces))
        faceHeight = np.zeros(shape=(nb_faces))
        for ind in range(nb_faces):
            faceWidth[ind]  = tb_face_rect[ind]['width']
            faceHeight[ind] = tb_face_rect[ind]['height']
        ind_max = np.argmax(faceWidth*faceHeight.T)

        global_var['age']     = tb_age[ind_max]
        global_var['gender']  = tb_gender[ind_max]
        global_var['emo']     = tb_emo[ind_max]

        chrome_server2client(clientId, 'DONE')

        return tb_age, tb_gender, tb_glasses, tb_emo
    else:
        return 'N/A','N/A','N/A','N/A'

"""
Yield Face and Emotion API results
"""
def get_face_emotion_api_results(clientId):

    resp = detect_face_attributes(clientId)
    if (resp==1):

        print 'Calling APIs to retrieve facial and emotional attributes, please wait'
        tb_age, tb_gender, tb_glasses, tb_emo = retrieve_face_emotion_att(clientId)

        if ([tb_age, tb_gender, tb_glasses, tb_emo] != ['N/A','N/A','N/A','N/A']):
            # Translate emotion to french
            tb_emo_eng = ['happiness', 'sadness', 'surprise', 'anger', 'fear',
                          'contempt', 'disgust', 'neutral']
            tb_emo_correspond = ['joyeux', 'trist', 'surprise',
                                 'en colère', "d'avoir peur", ' mépris',
                                 'dégoût', 'neutre']

            # Translate glasses to french
            tb_glasses_eng = ['NoGlasses', 'ReadingGlasses',
                              'sunglasses', 'swimmingGoggles']
            tb_glasses_correspond = ['ne portez pas de lunettes',
                                     'portez des lunettes',
                                     'portez des lunettes de soleil',
                                     'portez des lunettes de natation']

            for ind in range(len(tb_age)):
                glasses_str = tb_glasses_correspond[tb_glasses_eng.index(tb_glasses[ind])]
                emo_str     = tb_emo_correspond[tb_emo_eng.index(tb_emo[ind])]
                textToSpeak = "Bonjour " + ('Monsieur' if tb_gender[ind] =='male' else 'Madame') + \
                    ", vous avez " + tb_age[ind].replace('.',',') + " ans, votre état d'émotion est " + emo_str + \
                    ", et vous " + glasses_str
                simple_message(clientId, textToSpeak)
        else:
            print 'Found no faces'
            simple_message(clientId, u'Désolé, aucun visage trouvé')

        time.sleep(0.5)

"""
Ask a name or id as a string
"""
def ask_name(clientId, flag):
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    global_var['text']  = ''
    global_var['text2'] = ''
    global_var['text3'] = "Donnez-moi votre identifiant, s'il vous plait !"

    if (flag):
        simple_message(clientId, global_var['text3'])

    while (global_var['respFromHTML']==""):
        pass
    res = global_var['respFromHTML']
    global_var['respFromHTML'] = ""
    return res

"""
Using Haar Cascade detector to detect faces from a grayscale image
"""
def detect_faces(faceCascade, gray):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (50, 50),
        flags        = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces


"""
Get all images in database alongside with their labels
"""
def get_images_and_labels(path, list_nom):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = [] # images will contains face images
    labels = [] # labels which are assigned to the image

    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Get the label of the image
        nom = os.path.split(image_path)[1].split(".")[0]
        if nom not in list_nom:
            list_nom.append(nom)

        nbr = list_nom.index(nom) + 1

        images.append(gray)
        labels.append(nbr)

    # return the images list and labels list
    return images, labels



"""
==============================================================================
Dialogue from Chrome and Pepper
==============================================================================
"""

def chrome_server2client(clientId, text): # Text-to-Speech

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['sendToHTML'] = text
    time.sleep(0.1)

def chrome_client2server(clientId): # Speech-to-Text

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['respFromHTML'] = ''

    while (not (global_var['startListening']) and (global_var['respFromHTML'] == '')):
        pass
    global_var['startListening'] = False # Reset after using
    t0 = time.time() # Start counting time

    while (global_var['respFromHTML'] == ''):
        pass
        if (time.time()-t0>=10 and global_var['respFromHTML'] == ''): # Time outs after 7 secs
            global_var['respFromHTML'] = '@' # Silence

    resp = global_var['respFromHTML']
    time.sleep(0.1)
    global_var['respFromHTML'] = '' # Rewrite '' after getting result
    return resp

def chrome_yes_or_no(clientId, question):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    #chrome_server2client(clientId, question) # Ask a question
    app_pepper.pepper_tts(clientId, question) # Pepper asks a question
    response = chrome_client2server(clientId) # Wait for an answer

    if (response == '@'):
        result, response = chrome_yes_or_no(clientId, u"Je ne vous entends pas, veuillez répéter")

    if (response=='oui' or response=='non'):
        responseYesOrNo = response
    else:
        classes = natural_language_classifier.classify('2374f9x68-nlc-1265', response)
        responseYesOrNo = classes["top_class"]

    if not(global_var['flag_quit']):
        if (responseYesOrNo=='oui'):
            result = 1
        elif (responseYesOrNo=='non'):
            result = 0
        elif (responseYesOrNo=='not_relevant'):
            result, response = chrome_yes_or_no(clientId, u"Votre réponse n'est pas pertinente, veuillez ré-répondre")
#    else:
#        result   = -1
#        responseYesOrNo = ''
    return result, response


"""
==============================================================================
Streaming Video: runs streaming video independently with other activities
==============================================================================
"""
def video_streaming(clientId):
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    time_origine = time.time()

    while True:
        frame = global_var['frameFromHTML'] # Get frame from HTML
        frame = cv2.flip(frame, 1) # Vertically flip frame
        global_var['key'] = cv2.waitKey(1)
        if (global_var['key'] == 27 or global_var['key2'] == 27): # wait for ESC key to exit
            cv2.destroyWindow('ClientId: ' + str(clientId) + ' - Video streaming')
            global_var['flag_quit'] = 1 # Use global_vars
            break

        """
        Face Detection part
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to a grayscale image
        faces = detect_faces(faceCascade, gray) # Detect faces on grayscale image

        """
        Recognition part
        """
        for (x, y, w, h) in faces:
            if (len(faces)>1): # Consider only the biggest face appears in the video
                w_vect = faces.T[2,:]
                h_vect = faces.T[3,:]
                x0, y0, w0, h0 = faces[np.argmax(w_vect*h_vect)]
            elif (len(faces)==1): # If there is only one face
                x0, y0, w0, h0 = faces[0]

            if not global_var['flag_disable_detection']:
                cv2.rectangle(frame, (x0, y0), (x0+w0, y0+h0), (25, 199, 247), 1) # Draw a rectangle around the biggest face
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (25, 199, 247), 1) # Draw a rectangle around the faces

            if (len(faces)>=1):
                global_var['image_save'] = gray[y0 : y0 + h0, x0 : x0 + w0]
                nbr_predicted, conf      = recognizer.predict(global_var['image_save']) # Predict function

                nom = list_nom[nbr_predicted-1] # Get resulting name

                if (conf < thres): # if recognizing distance is less than the predefined threshold -> FACE RECOGNIZED
                    if not global_var['flag_disable_detection']:
                        txt = nom + ', distance: ' + str(conf)
                        message_xy(frame, txt, x0, y0-5, 'w', 1)

                    global_var['tb_nb_times_recog'][nbr_predicted-1] = global_var['tb_nb_times_recog'][nbr_predicted-1] + 1 # Increase nb of recognize times

                message_xy(frame, global_var['age'],    x0+w0, y0, 'b', 1)
                message_xy(frame, global_var['gender'], x0+w0, y0+10, 'b', 1)
                message_xy(frame, global_var['emo'],    x0+w0, y0+20, 'b', 1)

        # End of For-loop

        # Texts to display on video
        count_time = time.time() - time_origine
        fps = count_fps()

        message(frame, "Time: " + str(count_time)[0:4], 0, 1, 'g', 2)
        message(frame, "FPS: "  + str(fps)[0:5],        0, 2, 'g', 1)
        message(frame, global_var['text'],  0, 3, 'g', 1)
        message(frame, global_var['text2'], 0, 4, 'g', 1)
        message(frame, global_var['text3'], 0, 5, 'g', 1)

        # Frame display
        cv2.imshow('ClientId: ' + str(clientId) + ' - Video streaming', frame)

    cv2.destroyWindow('ClientId: ' + str(clientId) + ' - Video streaming')


"""
Put Texts on frame to display on streaming video at a predefined position (row,column)
"""
def message(frame, text, col, line, color, thickness):
    height, width = frame.shape[:2]
    if (col==0): x = 10
    if (line==1):
        y = 20
    elif (line==2):
        y = 40
    elif (line==3):
        y = height-50
    elif (line==4):
        y = height-30
    elif (line==5):
        y = height-10
    message_xy(frame, text, x, y, color, thickness)

"""
Put texts on frame to display on streaming video at position (x,y)
"""
def message_xy(frame, text, x, y, color, thickness):
    if color=='r':
        rgb = (0, 0, 255)
    elif color=='g':
        rgb = (0, 255, 0)
    elif color=='b':
        rgb = (255, 0, 0)
    elif color=='w':
        rgb = (255, 255, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, rgb, thickness, lineType=cv2.CV_AA)


"""
Display Formation Panel for a recognized or username-known user
"""
def go_to_formation(clientId, xls_filename, name):
    resp = ask_go_to_formation(clientId)
    if (resp==1):

        global global_vars
        global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

        global_var['age']     = ''
        global_var['gender']  = ''
        global_var['emo']     = ''

        global_var['flag_disable_detection'] = 1 # Disable the detection when entering Formation page
        global_var['flag_enable_recog']      = 0

        tb_formation = read_xls(xls_filename, 0) # Read Excel file which contains Formation info
        mail = reform_username(name) # Find email from name
        global_var['text'] = "Bonjour " + str(name)

        if (mail == '.'):
            global_var['text2'] = "Votre information n'est pas disponible !"
            global_var['text3'] = "Veuillez contacter contact@orange.com"
        else:
            mail_idx = tb_formation[0][:].index('Mail')

            # Get mail list
            mail_list = []
            for idx in range(0, len(tb_formation)):
                mail_list.append(tb_formation[idx][mail_idx])

            ind = mail_list.index(mail) # Find user in xls file based on his/her mail
            date = xlrd.xldate_as_tuple(tb_formation[ind][tb_formation[0][:].index('Date du jour')],0)
            text2 = "Bienvenue à la formation de "+str(tb_formation[ind][tb_formation[0][:].index('Prenom')])+" "+str(tb_formation[ind][tb_formation[0][:].index('Nom')] + ' !')
            text3 = "Vous avez un cours de " + str(tb_formation[ind][tb_formation[0][:].index('Formation')]) + ", dans la salle " + str(tb_formation[ind][tb_formation[0][:].index('Salle')]) + ", à partir du " + "{}/{}/{}".format(str(date[2]), str(date[1]),str(date[0]))
            global_var['text2'] = replace_accents2(text2)
            global_var['text3'] = replace_accents2(text3)

        simple_message(clientId, text2 + ' ' + text3)
        time.sleep(1)

        link='<a href="http://centre-formation-orange.mybluemix.net">ici</a>'
        chrome_server2client(clientId, u"SILENT Cliquez " + link + u" pour accéder à la page Formation pour plus d'information")
        time.sleep(0.5)

        cv2.waitKey(5000)
        return_to_recog(clientId) # Return to recognition program immediately or 20 seconds before returning
    else:
        return_to_recog(clientId) # Return to recognition program immediately or 20 seconds before returning

"""
Return to recognition program after displaying Formation
"""
def return_to_recog(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    if not global_var['flag_quit']:
        
        resp_quit_formation = quit_formation(clientId)
        if (resp_quit_formation == 0):
            time.sleep(5) # wait for more 5 seconds before quitting

        global_var['flag_disable_detection']  = 0
        global_var['flag_enable_recog']       = 1
        global_var['flag_ask']                = 1
        global_var['flag_reidentify']         = 0



"""
Find valid username
"""
def reform_username(name):

    if (name=='huy' or name=='huy_new'):
        firstname    = 'thanhhuy'
        lastname     = 'nguyen'
        email_suffix = '@orange.com'

    elif (name=='cleblain'):
        firstname    = 'christian'
        lastname     = 'leblainvaux'
        email_suffix = '@orange.com'

    elif (name=='catherine' or name=='lemarquis'):
        firstname    = 'catherine'
        lastname     = 'lemarquis'
        email_suffix = '@orange.com'

    elif (name=='ionel'):
        firstname    = 'ionel'
        lastname     = 'tothezan'
        email_suffix = '@orange.com'

    else:
        firstname = ''
        lastname = ''
        email_suffix = ''

    mail = firstname + '.' + lastname + email_suffix
    return mail


"""
==============================================================================
Taking photos
==============================================================================
"""
def take_photos(clientId, step_time, flag_show_photos):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    name = ask_name(clientId, 1)

    image_to_paths = [imgPath+str(name)+"."+str(i)+suffix for i in range(nb_img_max)]

    if os.path.exists(imgPath+str(name)+".0"+suffix):
        print u"Les fichiers avec le nom " + str(name) + u" existent déjà"
        b = yes_or_no(clientId, u"Les fichiers avec le nom " + str(name) + u" existent déjà, écraser ces fichiers ?", 3)
        if (b==1):
            for image_del_path in image_to_paths:
                os.remove(image_del_path)
        elif (b==0):
            name = ask_name(clientId, 1)
            image_to_paths = [imgPath + str(name)+"."+str(i)+suffix for i in range(nb_img_max)]

    global_var['text']  = 'Prenant photos'
    global_var['text2'] = 'Veuillez patienter... '

    simple_message(clientId, global_var['text']+', '+global_var['text2'])

    nb_img = 0
    while (nb_img < nb_img_max):
        image_path = image_to_paths[nb_img]
        cv2.imwrite(image_path, global_var['image_save'])
        print "Enregistrer photo " + image_path + ", nb de photos prises : " + str(nb_img+1)
        global_var['text3'] = str(nb_img+1) + ' ont ete prises, reste a prendre : ' + str(nb_img_max-nb_img-1)
        nb_img += 1
        time.sleep(step_time)

    # Display photos that has just been taken
    if flag_show_photos:
        thread_show_photos = Thread(target = show_photos, args = (clientId, imgPath, name), name = 'thread_show_photos_'+clientId)
        thread_show_photos.start()

    time.sleep(0.5)

    # Allow to retake photos and validate after finish taking
    thread_retake_validate_photos = Thread(target = retake_validate_photos, args = (clientId, step_time, flag_show_photos, imgPath, name), name = 'thread_retake_validate_photos_'+clientId)
    thread_retake_validate_photos.start()


"""
Retaking and validating photos
"""
def retake_validate_photos(clientId, step_time, flag_show_photos, imgPath, name):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    # Ask users if they want to change photo(s) or validate them
    b = validate_photo(clientId)
    image_to_paths = [root_path+imgPath+str(name)+"."+str(j)+suffix for j in range(nb_img_max)]

    while (b==0):
        global_var['text3'] = "Veuillez repondre"
        simple_message(clientId, u"Veuillez répondre quelles photos que vous voulez changer ?")

        while (global_var['respFromHTML'] == ""):
            pass
        nb = global_var['respFromHTML']
        global_var['respFromHTML'] = ""

        if ('-' in nb):
            nb2 = ''
            for i in range(int(nb[0]), int(nb[2])+1):
                nb2 = nb2 + str(i)
            nb = nb2
        elif (nb=='*' or nb=='all'):
            nb=''
            for j in range(0, nb_img_max):
                nb = nb+str(j+1)
        elif any(nb[idx] in a for idx in range(0, len(nb))): # If there is any number in string
            nb2 = ''
            for j in range(0, len(nb)):
                if (nb[j] in a):
                    nb2 = nb2 + nb[j]
            nb = nb2
        else:
            print 'Fatal error: invalid response'
            nb = ''

        nb = str_replace_chars(nb, [',',';','.',' '], ['','','',''])

        if (nb!=""):
            str_nb = ""
            for j in range(0, len(nb)):
                if (j==len(nb)-1):
                    str_nb = str_nb + "'" + nb[j] + "'"
                else:
                    str_nb = str_nb + "'" + nb[j] + "', "

            simple_message(clientId, 'Vous souhaitez changer les photos: ' + str_nb + ' ?')

            global_var['text']  = 'Prenant photos'
            global_var['text2'] = 'Veuillez patienter... '
            global_var['text3'] = ''

        for j in range(0, len(nb)):
            global_var['text3'] = str(j) + ' ont ete prises, reste a prendre : ' + str(len(nb)-j)
            time.sleep(step_time)
            print "Reprendre photo ", nb[j]
            image_path = image_to_paths[int(nb[j])-1]
            os.remove(image_path) # Remove old image
            cv2.imwrite(image_path, global_var['image_save'])
            print "Enregistrer photo " + image_path + ", nb de photos prises : " + nb[j]

        a = yes_or_no(clientId, u'Reprise de photos finie, souhaitez-vous réviser vos photos ?', 4)
        if (a==1):
            thread_show_photos2 = Thread(target = show_photos, args = (clientId, imgPath, name), name = 'thread_show_photos2_'+clientId)
            thread_show_photos2.start()

        b = validate_photo(clientId)
        global_var['text']  = ''
        global_var['text2'] = ''
        global_var['text3'] = ''
        if (b==1):
            break
    # End of While(b==0)

    # Update recognizer after taking and validating photos
    images, labels = get_images_and_labels(imgPath, list_nom)
    recognizer.update(images, np.array(labels))
    print u"Recognizer a été mis a jour..."

    global_var['flag_enable_recog'] = 1  # Re-enable recognition
    global_var['flag_ask'] = 1 # Reset asking


"""
Display photos that have just been taken, close them if after 5 seconds or press any key
"""
def show_photos(clientId, imgPath, name):
    x = 100; y = 600

    image_to_paths = [root_path + imgPath + str(name) + "." + str(j) + suffix for j in range(nb_img_max)]

    ind = 1
    for img_path in image_to_paths:
        #print img_path
        img = cv2.imread(img_path)
        cv2.imshow('clientId '+clientId+' - Photo '+str(ind), img)
        height, width = img.shape[:2]
        cv2.moveWindow('clientId '+clientId+' - Photo '+str(ind), x, y)
        x   += width
        ind += 1

    cv2.waitKey(7000) # wait a key for 7 seconds
    for ind in range(nb_img_max):
        cv2.destroyWindow('clientId '+clientId+' - Photo '+str(ind+1))

"""
==============================================================================
Re-identification: when a user is not recognized or not correctly recognized
==============================================================================
"""
def re_identification(clientId, nb_time_max, name0):

    simple_message(clientId, u'Veuillez rapprocher vers la camera, ou bouger votre tête...')

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    global_var['text']  = ''
    global_var['text2'] = ''
    global_var['text3'] = ''
    
    tb_old_name    = np.chararray(shape=(nb_time_max+1), itemsize=10) # All of the old recognition results, which are wrong
    tb_old_name[:] = ''
    tb_old_name[0] = name0

    nb_time = 0
    global_var['flag_enable_recog'] = 1
    global_var['flag_reidentify']   = 1
    global_var['flag_ask'] = 0

    while (nb_time < nb_time_max):
        time.sleep(wait_time) # wait until after the re-identification is done
        name1 = global_var['nom'] # New result

        if np.all(tb_old_name != name1) and global_var['flag_recog']:
            print 'Essaie ' + str(nb_time+1) + ': reconnu comme ' + str(name1)

            resp = validate_recognition(clientId, str(name1))
            print resp
            if (resp == 1):
                result = 1
                name = name1
                break
            else:
                result = 0
                nb_time += 1
                tb_old_name[nb_time] = name1

        elif (not global_var['flag_recog']):
            print 'Essaie ' + str(nb_time+1) + ': personne inconnue'
            result = 0
            nb_time += 1

    if (result==1): # User confirms that the recognition is correct now
        global_var['flag_enable_recog'] = 0
        # global_var['flag_reidentify']   = 0
        global_var['flag_wrong_recog']  = 0

        get_face_emotion_api_results(clientId)
        time.sleep(2)
        # global_var['text'], global_var['text2'], global_var['text3'] = go_to_formation(clientId, xls_filename, name)
        go_to_formation(clientId, xls_filename, name)
        # return_to_recog(clientId) # Return to recognition program immediately or 20 seconds before returning

    else: # Two time failed to recognized
        global_var['flag_enable_recog'] = 0 # Disable recognition when two tries have failed
        # global_var['flag_reidentify']   = 0
        simple_message(clientId, u'Désolé je vous reconnaît pas, veuillez me donner votre identifiant')

        name = ask_name(clientId, 0)
        if os.path.exists(imgPath+str(name)+".0"+suffix): # Assume that user's face-database exists if the photo 0.png exists
            simple_message(clientId, 'Bonjour '+ str(name)+', je vous conseille de changer vos photos')
            flag_show_photos = 1
            step_time = 1

            thread_show_photos3 = Thread(target = show_photos, args = (clientId, imgPath, name), name = 'thread_show_photos3_'+clientId)
            thread_show_photos3.start()

            time.sleep(0.5)
            thread_retake_validate_photos2 = Thread(target = retake_validate_photos, args = (clientId, step_time, flag_show_photos, imgPath, name), name = 'thread_retake_validate_photos2_'+clientId)
            thread_retake_validate_photos2.start()
        else:
            simple_message(clientId, "Malheureusement, les photos correspondant au nom " + str(name) + " n'existent pas. Je vous conseille de reprendre vos photos")

            time.sleep(1)
            global_var['flag_take_photo']  = 1  # Enable photo taking

    global_var['flag_reidentify']   = 0


"""
==============================================================================
Main program body with decision and redirection
==============================================================================
"""
def run_program(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    # Autorisation to begin Streaming Video
    optin0 = allow_streaming_video(clientId)

    if (optin0 == 1):
        global_var['key']       = 0
        global_var['flag_quit'] = 0

        # Thread of streaming video
        thread_video = Thread(target = video_streaming, args = (clientId,), name = 'thread_video_' + clientId)
        thread_video.start()

        start_time   = time.time() # For recognition timer (will reset after each 3 secs)
        time_origine = time.time() # For display (unchanged)

        """
        Permanent loop
        """
        i=0
        j=0
        while True:

            # Break While-loop and quit program as soon as the Esc key is pressed
            if (global_var['key'] == 27):
                break

            """
            Decision part
            """
            if not (global_var['flag_quit']): #TODO: new
                global_var['key2'] = cv2.waitKey(1)
                if (global_var['key2'] == 27):
                    break
                elapsed_time = time.time() - start_time
                if ((elapsed_time > wait_time) and global_var['flag_enable_recog']): # Identify after each 3 seconds
                    if (max(global_var['tb_nb_times_recog']) >= nb_max_times/2): # If the number of times recognized is big enough
                        global_var['flag_recog']  = 1 # Known Person
                        global_var['flag_ask']    = 0
                        global_var['nom'] = list_nom[np.argmax(global_var['tb_nb_times_recog'])] # Get name of recognizing face
                        global_var['text']  = 'Reconnu : ' + global_var['nom']

                        if (not global_var['flag_reidentify']):
                            global_var['text2'] = "Appuyez [Y] si c'est bien vous"
                            global_var['text3'] = "Appuyez [N] si ce n'est pas vous"

                            res_verify_recog = verify_recog(clientId, global_var['nom'])
                            if (res_verify_recog==1):
                                global_var['key'] = ord('y')
                            elif (res_verify_recog==0):
                                global_var['key'] = ord('n')

                    else: # If the number of times recognized anyone from database is too low
                        global_var['flag_recog'] = 0 # Unknown Person
                        global_var['nom']   = '@' # '' is for unknown person
                        global_var['text']  = 'Personne inconnue'
                        global_var['text2'] = ''
                        global_var['text3'] = ''

                        if (not global_var['flag_reidentify']):
                            global_var['flag_ask'] = 1
                            simple_message(clientId, u'Désolé, je ne vous reconnaît pas')
                            time.sleep(0.25)

                    global_var['tb_nb_times_recog'].fill(0) # reinitialize with all zeros

                    start_time = time.time()  # reset timer

                """
                Redirecting user based on recognition result and user's status (already took photos or not) in database
                """
                count_time = time.time() - time_origine
                if (count_time <= wait_time):
                    global_var['text3'] = 'Initialisation (pret dans ' + str(wait_time-count_time)[0:4] + ' secondes)...'
                    if i==0:
                        chrome_server2client(clientId, 'START')
                        i=1
                    if (global_var['flag_quit']):
                        break
                else:
                    """
                    Start Redirecting after the first 3 seconds
                    """
                    if (global_var['flag_quit']):
                        break
                    if j==0:
                        chrome_server2client(clientId, 'DONE')
                        j=1
                    if (global_var['flag_recog']):
                        if (global_var['key']==ord('y') or global_var['key']==ord('Y')): # User chooses Y to go to Formation page
                            global_var['flag_wrong_recog']  = 0
                            get_face_emotion_api_results(clientId)
                            time.sleep(1)
                            # global_var['text'], global_var['text2'], global_var['text3'] = go_to_formation(clientId, xls_filename, global_var['nom'])
                            go_to_formation(clientId, xls_filename, global_var['nom'])
                            global_var['key'] = 0
                            # return_to_recog(clientId) # Return to recognition program, after displaying Formation

                        if (global_var['key']==ord('n') or global_var['key']==ord('N')): # User confirms that the recognition result is wrong by choosing N
                            global_var['flag_wrong_recog'] = 1
                            global_var['flag_ask'] = 1
                            global_var['key'] = 0

                    if ((global_var['flag_recog'] and global_var['flag_wrong_recog']) or (not global_var['flag_recog'])): # Not recognized or not correctly recognized
                        if (global_var['flag_ask']):# and (not flag_quit)):

                            global_var['flag_enable_recog'] = 0 # Disable recognition in order not to recognize while re-identifying or taking photos

                            resp_deja_photos = deja_photos(clientId) # Ask user if he has already had a database of face photos

                            # if (resp_deja_photos==-1):
                            #     global_var['flag_ask'] = 0

                            if (resp_deja_photos==1): # User has a database of photos
                                # global_var['flag_enable_recog'] = 0 # Disable recognition in order not to recognize while re-identifying
                                global_var['flag_ask'] = 0

                                name0 = global_var['nom']   # Save the recognition result, which is wrong, in order to compare later
                                nb_time_max = 5             # Number of times to retry recognize

                                thread_reidentification = Thread(target = re_identification, args = (clientId, nb_time_max, name0), name = 'thread_reidentification_'+clientId)
                                thread_reidentification.start()

                            elif (resp_deja_photos == 0): # User doesnt have a database of photos

                                # global_var['flag_enable_recog'] = 0 # Disable recognition in order not to recognize while taking photos
                                resp_allow_take_photos = allow_take_photos(clientId)

                                if (resp_allow_take_photos==1): # User allows to take photos
                                    global_var['flag_take_photo'] = 1  # Enable photo taking
                                    #flag_enable_recog = 0 # Stop recognition while taking photos

                                else: # User doesnt want to take photos
                                    global_var['flag_take_photo'] = 0
                                    res = allow_go_to_formation_by_id(clientId)
                                    if (res==1): # User agrees to go to Formation in providing his id manually
                                        name = ask_name(clientId, 1)
                                        # global_var['text'], global_var['text2'], global_var['text3'] = go_to_formation(clientId, xls_filename, name)
                                        go_to_formation(clientId, xls_filename, name)

                                        # Return to recognition program (if user wishs to, otherwise, wait 20 seconds before returning anyway)
                                        # return_to_recog(clientId)

                                    else: # Quit if user refuses to provide manually his id (after all other functionalities)
                                        break

                                resp_allow_take_photos = 0
                            resp_deja_photos = 0
                        global_var['flag_ask'] = 0

                    if (global_var['flag_take_photo']):# and (not flag_quit)):

                        step_time  = 1 # Interval of time (in second) between two times of taking photo

                        thread_take_photo = Thread(target = take_photos, args = (clientId, step_time, 1), name = 'thread_take_photo_'+clientId)
                        thread_take_photo.start()

                        global_var['tb_nb_times_recog'] = np.empty(len(list_nom)+1) # Extend the list with one more value for the new face
                        global_var['tb_nb_times_recog'].fill(0) # reinitialize the table with all zeros
                        global_var['flag_take_photo'] = 0

                    """
                    Call Face API and Emotion API, and display
                    """
                    if (global_var['key']==ord('i') or global_var['key']==ord('I')):
                        retrieve_face_emotion_att(clientId)
                        global_var['key'] = 0
        """
        End of While-loop
        """
    """
    Exit the program
    """
    quit_program(clientId)


"""
Quit program
"""
def quit_program(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    global_var['flag_quit'] = 0 # Turn it on to execute just the yes_no question and bye-bye
    cv2.destroyWindow('ClientId: ' + str(clientId) + ' - Video streaming')

    #chrome_server2client(clientId, u"Merci de votre utilisation. Au revoir, à bientôt")
    app_pepper.pepper_tts(clientId, u"Merci de votre utilisation. Au revoir, à bientôt")
    global_var['flag_quit'] = 1


"""
Definition of used yes-no questions in program
"""
def ask_go_to_formation(clientId):
    resp = yes_or_no(clientId, u"Voulez-vous accéder votre page de Formation ?")
    return resp

def detect_face_attributes(clientId):
    resp = yes_or_no(clientId, "Voulez-vous apercevoir vos attributs faciales ?")
    return resp

def verify_recog(clientId, name):
    resp = yes_or_no(clientId, "Bonjour " + name + ", est-ce bien vous ?")
    return resp

def allow_streaming_video(clientId):
    resp = yes_or_no(clientId, 'Bonjour ! Voulez-vous lancer la reconnaissance faciale ?')
    return resp

def deja_photos(clientId):
    resp = yes_or_no(clientId, u'Avez-vous déjà pris des photos ?')
    return resp

def allow_take_photos(clientId):
    resp = yes_or_no(clientId, u"Êtes-vous d'accord pour vous faire prendre en photos ?")
    return resp

def validate_photo(clientId):
    resp = yes_or_no(clientId, 'Voulez-vous valider ces photos ?')
    return resp

def allow_go_to_formation_by_id(clientId):
    resp = yes_or_no(clientId, u"Voulez-vous accéder votre page Formation par votre identifiant ?")
    return resp

def quit_formation(clientId):
    resp = yes_or_no(clientId, 'Voulez-vous quitter la page Formation ?')
    return resp

def validate_recognition(clientId, name):
    resp = yes_or_no(clientId, "Bonjour " + name + ", est-ce bien vous cette fois-ci ?")
    return resp

"""
Yes/No question as an asking/answering by dialogue
"""
def yes_or_no(clientId, message):
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    if (not global_var['flag_quit']): # Put in If-condition to allow interrupt when Esc is pressed
        resp, ouinon = chrome_yes_or_no(clientId, message)
        return resp
    else:
        return -1

"""
Simple message as a notification speech
"""
def simple_message(clientId, message):
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    if (not global_var['flag_quit']): # Put in If-condition to allow interrupt when Esc is pressed
        #chrome_server2client(clientId, message)
        app_pepper.pepper_tts(clientId, message)

"""
Calculating Frame-per-second parameter
"""
def count_fps():
    # video = cv2.VideoCapture(-1)
    # fps   = video.get(cv2.cv.CV_CAP_PROP_FPS);
    # return fps
    return 'N/A'


"""
Initialisation for global variables used by clientId
"""
def global_var_init(clientId):

    global global_vars

    # Messages to appear on streaming video (at line 3, 4, 5) and attributes
    text    = ''
    text2   = ''
    text3   = ''
    age     = ''
    gender  = ''
    emo     = ''

    # Flags used in program
    flag_recog        = 0 # Recognition flag (flag=1 if recognize someone, flag=0 otherwise)
    flag_take_photo   = 0 # Flag if unknown user chooses to take photos
    flag_wrong_recog  = 0 # Flag if a person is recognized but not correctly, and feedbacks
    flag_enable_recog = 1 # Flag of enabling or not the recognition
    flag_disable_detection = 0 # Flag of disabling displaying the detection during some other task (Formation, Taking photos)
    flag_quit         = 0
    flag_ask          = 0 # Flag if it is necessary to ask 'etes vous dans ma base ou pas ?'
    flag_reidentify   = 0

    # Initialisation global variables
    image_save = 0
    key   = 0 # Quit key inside video streaming thread
    key2  = 0 # Quit key from run program

    sendToHTML = ''
    respFromHTML  = ""
    frameFromHTML = 0
    startListening = False

    tb_nb_times_recog = np.empty(len(list_nom))
    tb_nb_times_recog.fill(0) # initialize with all zeros

    nom = ''

    global_vars.append( dict([  ('clientId', str(clientId)),
                                ('text', text), ('text2', text2), ('text3', text3),
                                ('age', age), ('gender', gender), ('emo', emo),
                                ('flag_recog', flag_recog),
                                ('flag_take_photo', flag_take_photo),
                                ('flag_wrong_recog', flag_wrong_recog),
                                ('flag_enable_recog', flag_enable_recog),
                                ('flag_disable_detection', flag_disable_detection),
                                ('flag_quit', flag_quit),
                                ('flag_ask', flag_ask),
                                ('flag_reidentify', flag_reidentify),
                                # ('flag_pepper_start', flag_pepper_start),
                                ('image_save', image_save),
                                ('key', key),
                                ('key2', key2),
                                ('sendToHTML', sendToHTML),
                                ('respFromHTML', respFromHTML),
                                ('frameFromHTML', frameFromHTML),
                                ('startListening', startListening),
                                ('tb_nb_times_recog', tb_nb_times_recog),
                                ('nom', nom)
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
        return render_template('index_pepper.html')

    @app.route('/css/<path:filename>')
    def sendCSS(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'templates', 'css'), filename)

    @app.route('/javascript/<path:filename>')
    def sendJavascript(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'templates', 'javascript'), filename)

    @app.route('/images/<path:filename>')
    def sendImages(filename):
        return send_from_directory(os.path.join(os.getcwd(), 'templates', 'images'), filename)

    @app.route('/chat', methods=['POST'])
    def getResponseFromHTML():
        clientId = request.args.get('id')
        text     = request.args.get('text')
        if (text=='START'): # Start a conversation
            global_var_init(clientId)
            global flag_pepper_start
            flag_pepper_start = False

            # Pepper
            thread_pepper = Thread(target=run_app_pepper, args=(clientId,), name='pepper_'+str(clientId))
            thread_pepper.start()

            cv2.destroyAllWindows()

            while (not flag_pepper_start):
                time.sleep(0.5)
                print flag_pepper_start

            # Run program
            thread_program = Thread(target = run_program, args= (clientId,), name = 'thread_prog_'+clientId)
            thread_program.start()

        else: # Continue the conversation
            global global_vars
            global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()
            global_var['respFromHTML'] = text
        return "", 200

    @app.route('/startListening', methods=['POST'])
    def onstartListening():
        clientId = request.args.get('id')
        global global_vars
        global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
        global_var['startListening'] = True
        return "", 200

    @app.route('/wait', methods=['POST'])
    def waitForServerInput():
        clientId = request.args.get('id')
        time.sleep(0.5)
        global global_vars
        global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()
        temp        = global_var['sendToHTML']
        global_var['sendToHTML'] = ""
        return temp, 200

"""
Pepper
"""
class Pepper(object):
    def __init__(self):
        self.ip = "10.69.128.84"
        self.port = 9559
        self.session = qi.Session()
        try:
            self.session.connect("tcp://" + self.ip + ":" + str(self.port))
        except RuntimeError:
            print ( "Can't connect to Naoqi at ip \"" + self.ip + "\" on port " + str(self.port) +".\n"
                    "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)

        self.ALTextToSpeech = self.session.service('ALTextToSpeech')
        self.ALTextToSpeech.setLanguage('French')
        self.ALTextToSpeech.setParameter('speed', 110)

        self.ALVideoDevice = self.session.service('ALVideoDevice')
        self.ALVideoDevice.unsubscribe("CameraTop_0")
        self.ALVideoDevice.setParameter(0, 14, 2)
        self.handle = self.ALVideoDevice.subscribeCamera("CameraTop", 0, 2, 11, 5)

        self.ALTabletService = self.session.service("ALTabletService")

    def run_camera(self, clientId):
        global global_vars
        global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
        while True:
            self.ALVideoDevice.releaseImage(self.handle)
            data_pepper = self.ALVideoDevice.getImageRemote(self.handle)
            width, height, nbLayers = data_pepper[0:3] # from Documentation

            image_data = np.zeros((len(data_pepper[6]),1)) # data[6]: array of height*width*nbLayes containing image data
            data_bin = b2a_hex(str(data_pepper[6]))
            for k in range(0,len(data_pepper[6])):
                image_data[k] = int(data_bin[2*k:2*k+2], 16)
            image_reshape = np.reshape(image_data, (nbLayers, width, height), order='F')

            # data_uint8 = np.fromstring(data_pepper[6], np.uint8)
            # image_reshape = np.reshape(data_uint8, (nbLayers, width, height), order='F')

            imgRGB = np.dstack((image_reshape[2].T,image_reshape[1].T,image_reshape[0].T))
            cv2.imwrite('output.png', imgRGB)
            frame = cv2.imread('output.png')
            global_var['frameFromHTML'] = frame

    def pepper_tts(self, clientId, text):

        self.ALTextToSpeech.say(text)
        chrome_server2client(clientId, text)

        # Calculate the time needed to wait, until the TTS is finished
        text2 = str_replace_chars(text, [' ?',' !',' :',' ;'], ['?','!',':',';'])
        nbOfWords  = len(text2.split())
        timeNeeded = float(nbOfWords)/120*60 # Average words-per-min = 130
        time.sleep(timeNeeded)

    def pepper_show_images(self, clientId, imgName): #TODO: new function of showing images on Tablet
        self.ALTabletService.showImage("http://"+IP_address+"/face_database/"+imgName+suffix)
        time.sleep(3)
        self.ALTabletService.hideImage()

def run_app_pepper(clientId):
    global app_pepper, flag_pepper_start
    # global global_vars
    # global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    app_pepper = Pepper()
    # global_var['flag_pepper_start']= True
    flag_pepper_start = True
    app_pepper.run_camera(clientId)


"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
# Parameters
root_path    = ""
cascPath     = "haarcascade_frontalface_default.xml" # path to Haar-cascade training xml file
imgPath      = "face_database/" # path to database of faces
suffix       = '.png' # image file extention
thres        = 80     # Distance threshold for recognition
wait_time    = 2.5    # Time needed to wait for recognition
nb_max_times = 6     # Maximum number of times of good recognition counted in 3 seconds (manually determined, and depends on camera)
nb_img_max   = 5      # Number of photos needs to be taken for each user
xls_filename = 'formation.xls' # Excel file contains Formation information

# Haar cascade detector used for face detection
faceCascade = cv2.CascadeClassifier(root_path + cascPath)

# For face recognition we use the Local Binary Pattern Histogram (LBPH) Face Recognizer
recognizer  = cv2.createLBPHFaceRecognizer()
list_nom    = []

# Call the get_images_and_labels function and get the face images and the corresponding labels
print u"Obtenu Images et Labels à partir de database..."
images, labels = get_images_and_labels(root_path + imgPath, list_nom)

# Perform the training
recognizer.train(images, np.array(labels))
print u"Apprentissage a été fini...\n"

# Natural Language Classifier
natural_language_classifier = NaturalLanguageClassifierV1(
                              username = '82376208-a089-464c-a5da-96893ed1aa89',
                              password = 'SEuX8ielPiiJ')

global_vars = []

flask_init()
port = int(os.getenv('PORT', '9099'))
app.run(host = '0.0.0.0', port = port, threaded = True)
