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
* mel spectrogram (128dim) + RMSE (1dim) (vector augmentation)
### Architecture
* CNN + BiLSTM-Self attention (concatenation)
