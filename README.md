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
* CNN + BiLSTM-Self attention (concatenation)
* CNN<br/>
Conv (5 by 5, 32 filters, ReLU) &nbsp;- BN - MaxPool (2 by 2) - Dropout (0.3) >><br/>
Conv (5 by 5, 64 filters, ReLU) &nbsp;- BN - MaxPool (2 by 2) - Dropout (0.3) >><br/>
Conv (3 by 3, 128 filters, ReLU) - BN - MaxPool (2 by 2) - Dropout (0.3) >><br/>
Conv (3 by 3, 32 filters, ReLU) &nbsp;- BN - MaxPool (2 by 1) >><br/>
Conv (3 by 3, 32 filters, ReLU) &nbsp;- BN - MaxPool (2 by 1) >> Flatten
* BiLSTM-Self attention

