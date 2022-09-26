# Detection of neurodegenerative diseases by the automatic analysis of handwriting

## Abstract
Neurodegenerative diseases are a serious issue which encompasses a myriad of complex and incurable disorders. in this thesis, we focus on Parkinson’s disease (PD), specifically; the detection of PD though the automatic analysis of offline handwriting. To accomplish this task; we propose Park-Net, our own convolutional neural network (CNN) architecture. We proceed to test this CNN on three PD handwriting datasets before comparing the results to state-of-the-art works, and with a 98% accuracy, and to the best of our knowledge; Park-Net outperforms studies as recent as (2022).

## Data Sets used
In this work we made use of 3 freely available PD handwriting datasets: HandPD,
NewHandPD, and Parkinson’s drawing. with a consistant image size of 224x224.
 #### [get datasets (.jpg).](https://drive.google.com/drive/folders/1cU4ucbJW76xo5Eyo1DCIQHk0I2w1fse9?usp=sharing)
 #### [get datasets (serialized).](https://drive.google.com/drive/folders/1QAF8T_fkaqO8TGL3ilZqVIHVCOf52fHh?usp=sharing)

## [Models](https://drive.google.com/drive/folders/1PNGYk7iuq0CRqzqZsrMboHH7JBsVLDTV?usp=sharing)
all models were implemented using:
 Python 3.7.13 
 Tensorflow and Keras 2.8.0
trainings were done on the Google Colab (free) platform using the serealized DataSets.

### Checkpoints used:
 Checkval: monitors the validation accuracy attribute, saves the Max value (bestval.hdf5)
 Checkacc: monitors the training accuracy attribute, saves the Max value (bestacc.hdf5)
 Checkvloss: monitors the validation loss attribute, saves the Min value (bestvloss.hdf5)

### Early stop:
all models are given 100 as the max number of epochs,
monitored value for early stop is training accuracy with a patience of 5.

## GUI implementation
Simple implementation done using the tkinter library.
__
![pdd_logo](https://user-images.githubusercontent.com/71077535/192288994-a88397c6-edc4-4529-b8db-126d3ea87070.png)

