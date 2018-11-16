#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = (
    'detect',
    'post_process',
)

import requests
import json
import collections
import itertools
import math
import sys

import time

from picamera.array import PiRGBArray
from picamera import PiCamera

import cv2
import numpy
import tensorflow as tf

import common
import model
import yolo
from PIL import Image

def prRed(prt): print("\033[91m{}\033[00m".format(prt));

def prGreen(prt): print("\033[92m{}\033[00m".format(prt));

def prYellow(prt): print("\033[93m{}\033[00m".format(prt));

def prPurple(prt): print("\033[95m{}\033[00m".format(prt));

def prCyan(prt): print("\033[96m{}\033[00m".format(prt));
                      
def consulta_placa(placa):
    if (placa!=None):
        headersx = {"Accept-Encoding":"identity",
           "X-Titanium-Id":"db924f67-05e6-4652-9222-5f5a28dc18b4",
           "Content-Type":"application/json; charset=utf-8;",
           "X-Requested-With": "XMLHttpRequest",
           "User-Agent":"Appcelerator Titanium/7.0.1 (GT-I9505; Android API Level: 21; pt-BR;)",
           "Host":"online7.detran.pe.gov.br",
           "Connection":"Keep-Alive"};
    page = requests.get("http://online7.detran.pe.gov.br/WCFJsonVeiculo/JsonVeiculo.svc/ConsultaVeiculoDados/{}/PE".format(placa), headers=headersx)
    status_code = page.status_code;
    page = page.json();
    #print(page);
    page = page["ConsultaVeiculoDadosResult"];
    page = page[0];

    if status_code==200:
        if page["MensagemErro"]==None:
                lblSituacao = page["sSituacao"];
                lblPlacaRefletiva = page["PlacaRefletiva"];
                lblEspecie = page["sTipoVeiculo"];
                lblCor = page["sCorVeiculo"];
                lblChassi = page["sChassi"];
                lblCombustivel = page["sCombustivel"];
                lblMarcaModelo = page["sMarcaVeiculo"];
                lblAnoFab = page["sAnoFabricacao"];
                lblAnoModelo = page["sAnoModelo"];
                lblPot = page["sPotencia"];
                lblCilindrada = page["sCilindrada"];
                lblCategoria = page["sCategoriaVeiculo"];
                lblCapPassag = page["sCapPassag"];
                lblRestricao1 = page["sNomeRestr1"];
                lblRestricao2 = page["sNomeRestr2"];
                lblRestricao3 = page["sNomeRestr3"];
                lblRestricao4 = page["sNomeRestr4"];
                lblCRV = page["sNumCrv"];
                prYellow("SITUAÇÃO: {}".format(lblSituacao));
                prYellow("{}".format(lblPlacaRefletiva));
                prYellow("ESPECIE: {}".format(lblEspecie));
                prYellow("COR: {}".format(lblCor));
                prYellow("CHASSI: {}".format(lblChassi));
                prYellow("COMBUSTÍVEL: {}".format(lblCombustivel));
                prYellow("MARCA/MODELO: {}".format(lblMarcaModelo));
                prYellow("ANO FAB: {}".format(lblAnoFab));
                prYellow("ANO MODELO: {}".format(lblAnoModelo));
                prYellow("POTENCIA: {}".format(lblPot));
                prYellow("CILINDRADA: {}".format(lblCilindrada));
                prYellow("CATEGORIA: {}".format(lblCategoria));
                prYellow("CAPACIDADE: {}".format(lblCapPassag));
                prYellow("RESTRICAO 1: {}".format(lblRestricao1));
                prYellow("RESTRICAO 2: {}".format(lblRestricao2));
                prYellow("RESTRICAO 3: {}".format(lblRestricao3));
                prYellow("RESTRICAO 4: {}".format(lblRestricao4));
        else:
            print(page["MensagemErro"]);
        

def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


if __name__ == "__main__":
    weights_cnn = 'weights_20180708_171720_feedback.npz';
    weights_yolo = 'model_data/yolo_19000.h5';
    start_time = time.time();
    prYellow('Carregando Yolov3...');
    yolo = yolo.YOLO(weights_yolo);
    prGreen('Yolov3: modelo, âncoras e classes carregadas.');
    prYellow('Carregando Rede Neural Convolucional (CNN)...');
    f = numpy.load(weights_cnn);
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    x, y, params = model.get_training_model()
    prGreen('CNN: Modelo e pesos carregados.');
 
    time.sleep(2)
    prYellow('Inicializando sessão do Tensorflow...');
    with tf.Session(config=tf.ConfigProto()) as sess:
        feed_dict = {x: numpy.zeros([1,64,128])}
        feed_dict.update(dict(zip(params, param_vals)))
        y_val = sess.run(y, feed_dict=feed_dict)
        out_boxes, out_probs, out_labels = yolo.get_detections(Image.fromarray(numpy.zeros([100,100,3]).astype('uint8')));
        prGreen('Sessão do Tensorflow inicializada.');
        prYellow('Inicializando câmera...');
        camera = PiCamera()
        camera.resolution = (1024, 768)
        camera.framerate = 16 #original 32
        camera.shutter_speed = 60000000 #60000000
        camera.iso = 1600 #original 1600
        rawCapture = PiRGBArray(camera, size=(1024, 768))
        print("----- Carregado em %s segundos ------" % (time.time() - start_time));

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ##        cv2.rectangle(image, (320-128, 240-64), (320+128, 320+64), (255,0,0), 2)
            #image = image[240-128:240+128, 320-256:320+256]
            # show the frame
            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF
            
            rawCapture.truncate(0)

            if key == ord("p"):
                print('Processando...');
                start_time = time.time()
                im = frame.array
                #im = im[240-128:240+128, 320-256:320+256]
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #im = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA);
                
                #im = im / 255.
                #im = cv2.GaussianBlur(im, (1,1),0);
                #im = cv2.equalizeHist(im) / 255.;
                #im += numpy.random.normal(scale=0.05, size = im.shape);
                #im = numpy.clip(im, 0., 1.);
                #cv2.imshow("Frame2", im)
                im_pil = Image.fromarray(im);
                out_boxes, out_probs, out_labels = yolo.get_detections(im_pil);
                print('{} caixa(s) para a imagem.'.format(len(out_boxes)))
                for box in out_boxes:
                    top,left,bottom,right = box;
                    #cv2.rectangle(im, (left, top), (right, bottom), (0,255,0), 2);
                cv2.imshow("Frame", im)
                cv2.waitKey(1);
                for i,c in reversed(list(enumerate(out_labels))):
                    box = out_boxes[i];
                    score = out_probs[i];
                    top,left,bottom,right = box;
                    
                    scale = 0.15;
                    left = left - scale*(right-left);
                    right = right + scale*(right-left);
                    top = top - scale*(bottom-top);
                    bottom = bottom + scale*(bottom-top);

                    top = top - ((right-left) - 2*(bottom-top))/4;
                    bottom = bottom + ((right-left) - 2*(bottom-top))/4;

                    top = max(0,numpy.round(top).astype('int32'));
                    left = max(0,numpy.round(left).astype('int32'));
                    bottom = min(im_pil.size[1],numpy.round(bottom).astype('int32'));
                    right = min(im_pil.size[0],numpy.round(right).astype('int32'));
                    
    ##                cv2.rectangle(im, (left, top), (right, bottom), (255,0,0), 2);
    ##                cv2.imshow("Frame2", im)

                    im_crop = im[top:bottom, left:right];
                    im_crop = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY);
                    im_crop = cv2.resize(im_crop, (128,64), interpolation = cv2.INTER_CUBIC);
                    #im_crop = cv2.GaussianBlur(im_crop, (1,1),0);
                    #im_crop = cv2.equalizeHist(im_crop) / 255.;
                    #cv2.imshow("Frame2", im_crop)
                    im_crop = im_crop / 255.
                    im_crop2 = im_crop.copy();
                    #print("----- %s seconds ------" % (time.time() - start_time));
                    im_crop = im_crop.reshape(1,64,128);
                    feed_dict = {x: im_crop}
                    feed_dict.update(dict(zip(params, param_vals)))
                    y_val = sess.run(y, feed_dict=feed_dict)
                    present_prob = common.sigmoid(y_val[0,0])
                    letter_probs = (y_val[0, 1:].reshape(7, len(common.CHARS)))
                    letter_probs = common.softmax(letter_probs)
                    code = letter_probs_to_code(letter_probs);
                    prPurple('Caixa: {}'.format(box));
                    prPurple('Detecção (Yolo): {}'.format(score));
                    prPurple('Detecção (CNN): {}'.format(present_prob));
                    prCyan('Placa: {}'.format(code));
                    consulta_placa(code);
                    #print('-------------------------------');
                    cv2.rectangle(im, (left, top), (right, bottom), (0,0,255), 2);
                    cv2.putText(im,
                                code,
                                (left+5,top+30),
                                cv2.FONT_HERSHEY_PLAIN, 
                                2,
                                (0, 0, 0),
                                thickness=7)

                    cv2.putText(im,
                                code,
                                (left+5,top+30),
                                cv2.FONT_HERSHEY_PLAIN, 
                                2,
                                (255, 255, 255),
                                thickness=4)
                print("----- %s segundos ------" % (time.time() - start_time));
                cv2.imshow("Frame", im)
                cv2.waitKey(0)
                #camera.annotate_text = code;
        
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            if key == ord('c'):
                print('\n'*100);
