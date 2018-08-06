from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, SimpleRNN, Input, Conv2D, MaxPooling2D, Flatten, RepeatVector, Reshape, BatchNormalization as BN

import config as cf


def encoder_model(inputs):
    
    x = Conv2D(32, (3,3), padding='same', activation='tanh', name='conv1')(inputs)
    x = BN(name='bn1')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(64, (3,3), padding='same', activation='tanh', name='conv2')(x)
    x = BN(name='bn2')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(256, (3,3), padding='same', activation='tanh', name='conv3')(x)
    x = BN(name='bn3')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    x = RepeatVector(n=1)(x)
    
    x, ES_h, ES_c = LSTM(cf.Latent_dim, return_state=True, name='encoder')(x)

    return ES_h, ES_c


def decoder_model():
    
    dec_inputs = Input(shape=(None, cf.Vocabrary_num), name='decoder_input')
    dec_lstm = LSTM(cf.Latent_dim, return_sequences=True, return_state=True, name='decoder')
    dec_dense = Dense(cf.Vocabrary_num, activation='softmax', name='decoder_output')

    return dec_inputs, dec_lstm, dec_dense


# Training model
def model_train():
    inputs = Input(shape=(cf.Height, cf.Width, 3), name='encoder_input')
    ES_h, ES_c = encoder_model(inputs)
    dec_inputs, dec_lstm, dec_dense = decoder_model()
    dec_outputs, _, _ = dec_lstm(dec_inputs, initial_state=[ES_h, ES_c])
    dec_outputs = dec_dense(dec_outputs)

    model = Model(inputs=[inputs, dec_inputs], outputs=[dec_outputs])
    return model


# Testing model
def model_test_encoder():
    inputs = Input(shape=(cf.Height, cf.Width, 3), name='encoder_input')
    ES_h, ES_c = encoder_model(inputs)
    model = Model(inputs=[inputs], outputs=[ES_h, ES_c])
    return model


def model_test_decoder():
    dec_inputs, dec_lstm, dec_dense = decoder_model()

    DSI_h = Input(shape=(cf.Latent_dim,))
    DSI_c = Input(shape=(cf.Latent_dim,))
    dec_outputs, DS_h, DS_c = dec_lstm(dec_inputs, initial_state=[DSI_h, DSI_c])
    dec_outputs = dec_dense(dec_outputs)
    
    model_in = [dec_inputs] + [DSI_h, DSI_c]
    model_out = [dec_outputs] + [DS_h, DS_c]
    model = Model(inputs=model_in, outputs=model_out)
    
    return model
