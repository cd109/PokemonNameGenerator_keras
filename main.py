#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
#-- GPU 
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import numpy as np
import random
import argparse
import cv2
from glob import glob

import config as cf
from data_loader import DataLoader, get_ids
from model import *


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    

## for training function
def train():

    print('Training Start!')

    model = model_train()

    optimizer = keras.optimizers.Nadam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    ## Prepare Training data
    dl_train = DataLoader(phase='Train', shuffle=True)

    for epoch in range(cf.Epoch):
        epoch += 1
        x, decoder_x, decoder_y = dl_train.get_minibatch(shuffle=True)

        history = model.train_on_batch(x={'encoder_input':x, 'decoder_input': decoder_x},
                                       y={'decoder_output':decoder_y})
        
        train_loss = history[0]
        train_acc = history[1]

        if epoch % 10 == 0:
            print("Epoch: {}, Train_loss: {}, Train_accuracy: {}".format(
                epoch, train_loss, train_acc))
            
        
    model.save_weights(cf.Save_path)
    print('Trained model saved -> {}'.format(cf.Save_path))


## for test function
def test():
    
    print('Test start!')
    
    encoder_model = model_test_encoder()
    encoder_model.load_weights(cf.Save_path, by_name=True)

    decoder_model = model_test_decoder()
    decoder_model.load_weights(cf.Save_path, by_name=True)

    voca = cf.Vocabrary
    ids = get_ids()

    img_paths = get_image_paths()

    for img_path in img_paths:
        x = cv2.imread(img_path)
        x = cv2.resize(x, (cf.Width, cf.Height))
        x = x[..., ::-1]
        x = x[None, ...].astype(np.float32)
        x /= 255.

        p_id = os.path.basename(img_path).split('_')[0]
        gt = ids[p_id]

        states = encoder_model.predict_on_batch(x)
        
        inputs = np.zeros((1, 1, cf.Vocabrary_num), dtype=np.float32)
        inputs[0, 0, voca.index(cf.EOS)] = 1.

        out_name = ''
    
        for _ in range(cf.Name_length):
            
            outputs = decoder_model.predict_on_batch([inputs] + states)
            y = outputs[0]
            y = y[0, 0]
            states = outputs[1:]

            pred_char = voca[y.argmax()]
            if pred_char == cf.EOS:
                break

            inputs = np.zeros((1, 1, cf.Vocabrary_num), dtype=np.float32)
            inputs[0, 0, voca.index(pred_char)] = 1.
        
            out_name += pred_char
        
        print(img_path, gt, out_name)



def get_image_paths():
    img_paths = []

    for dir_path in cf.Test_dirs:
        for ext in cf.File_extensions:
            img_paths += glob(dir_path + '/*{}'.format(ext))

    img_paths.sort()
    return img_paths
    

def parse_args():
    parser = argparse.ArgumentParser(description='OsarePoem-Generator demo')
    parser.add_argument('--train', dest='train', help='train', action='store_true')
    parser.add_argument('--test', dest='test', help='test', action='store_true')
    
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':

    args = parse_args()
    
    check_dir(cf.Save_directory)

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
    
