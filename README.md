# KorInto
5-class sentence-final intonation classifier for a syllable-timed and head-final language (Korean)

## Requirements
librosa, Keras (TensorFlow), Numpy

## Data annotation
Manual tagging on the recordings of Korean drama scripts (# instances: 7,000)
### Classification into five categories
<img src="https://github.com/warnikchow/korinto/blob/master/fig2.png" width="500">

## System Description
### Feature (last 300 frames)
* mel spectrogram (128dim) + RMSE (1dim) (vector augmentation) >> (300 x 129)
* FFT window: 2,048
* Hop length: 512

### Architecture
* (CNN + BiLSTM-Self attention) (concatenation) >> MLP
* CNN<br/>
Conv (5 by 5, 32 filters, ReLU) - BN - MaxPool (2 by 2) - Dropout (0.3) >><br/>
Conv (5 by 5, 64 filters, ReLU) - BN - MaxPool (2 by 2) - Dropout (0.3) >><br/>
Conv (3 by 3, 128 filters, ReLU) - BN - MaxPool (2 by 2) - Dropout (0.3) >><br/>
Conv (3 by 3, 32 filters, ReLU) - BN - MaxPool (2 by 1) >><br/>
Conv (3 by 3, 32 filters, ReLU) - BN - MaxPool (2 by 1) >> Flatten
* BiLSTM-Self attention<br/>
BiLSTM hidden layer sequence: (300, 64)<br/>
Attention source: np.zeros(64)<br/>
Attention source >> Dense(64, ReLU) >> Context vector (64)<br/>
Context vector x BiLSTM hidden layer sequence (column-wisely) >> Attention vector (300,1)<br/>
Attention vector x BiLSTM hidden layer sequence (column-wisely) >> Weighted hidden layers (300,64)<br/>
* MLP<br/>
(CNN + BiLSTM Self-attention) >> Dense(64, ReLU) - Dropout (0.3) >> Dense(64, ReLU) - Dropout (0.3) >> Softmax(5)
