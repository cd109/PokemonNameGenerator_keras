## Config setting

import os

Voca = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョッー@"

Vocabrary = [x for x in Voca]
Voca_num = len(Vocabrary)

# Data path
Vocabrary_path = 'katakana.txt'
gt_file_path = 'pokemon.txt'

# LSTM parameters
Name_length = 5
EOS = '@'
Name_length += 2

## LSTM's latent dimension
Latent_dim = 128

# CNN parameters
## input size
Height = 128
Width = 128


# Training
Minibatch = 10
Epoch = 100
Learning_rate = 0.01
Weight_decay = 0.0005

## Data augmentation
Horizontal_flip = True
Vertical_flip = True
Rotate_ccw90 = False

## Directory paths for training
Train_dirs = [
    'Data/',
]

## file name extensions
File_extensions = ['.jpg', '.png']

# Test
Test_dirs = [
    'Data/',
]

Save_directory = 'output'
Save_model = 'PNG.h5'
Save_path = os.path.join(Save_directory, Save_model)

# other config
Random_seed = 0


# vocabrary_num contains EOS(@)
with open(Vocabrary_path, 'r') as f:
    Vocabrary_num = len(f.readlines()) + 1
