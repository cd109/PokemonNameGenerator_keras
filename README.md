# PokemonNameGenerator_keras
Pokemo Name Generator implemented with Keras-2.2.0.

This requires any pokemon image as input, outputs pokemon name.

Output is independing on dictionary, so this outputs any name.

This contains Encoder(CNN+LSTM), and Decoder(LSTM + Dense).

I confirmed using "ピカチュウ(Pikachu, id:25)", "ボーマンダ(Salimence, id:373)" and "ガブリアス(Garchomp, id:445)".

This is motivated by https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

# Requirements

```
Python-3.6.4
tensorflow-1.8.0
keras-2.2.0
opencv-python-3.4.1.15
numpy-1.14.0
OpenCV-3.4
```

# Setting
You cat set files and datasets as described below.

You can set image file name as "{id}_{num}.jpg(png)".

"id" is official pokemon id, and "num" is any number which is regardless of any setting.

e.g.) Pikachu -> 25_1.jpg, 25_2.jpg, 25_3.png, ...
      

```
PokemonNameGenerator_keras --- Data --- *.jpg(or png)
                            |- output - PNG.h5
                            |
                            |- config.py
                            |- data_loader.py
			    |- main.py
                            |- model.py
			    |- pokemon.txt
```

# Training
When training, you change config.py responding to your environment.


If you train with your original dataset, please type below command.

```
python main.py --train
```

Trained model is stored in "output/png.h5" .
You can change this pass using config.py .

You can use data-augmentation using config.py .

# Test
Whene testing, you change config.py.

The datails are as Training.

If you test with your original dataset, please type below command.

```
python main.py --test
```

# config.py
You can change class_labels, dataset paths, hyper-parameters(minibatch, learning rate and so on),
data augmentation.

If you want to use horizontal, vertical flip or rotation in training, you change "Horizontal_flip", "Vertical_flip" or "Rotate_ccw90" to "True" from "False".
