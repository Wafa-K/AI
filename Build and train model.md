## Skeleton Code

The code below provides a skeleton for the model building & training component of your project. You can add/remove/build on code however you see fit, this is meant as a starting point.


```python
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
#%matplotlib inline
import matplotlib.pyplot as plt
from itertools import chain
import seaborn as sns
from random import sample
import scipy
import tensorflow as tf
from skimage import io

import sklearn as sk
import sklearn.model_selection as skl
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, plot_precision_recall_curve, f1_score, confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

##Import any other stats/DL/ML packages you may need here. E.g. Keras, scikit-learn, etc.
```

    Using TensorFlow backend.


## Do some early processing of your metadata for easier model training:


```python
## Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation
## Load the NIH data to all_xray_df
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)
```

    Scans found: 112120 , Total Headers 112120





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image Index</th>
      <th>Finding Labels</th>
      <th>Follow-up #</th>
      <th>Patient ID</th>
      <th>Patient Age</th>
      <th>Patient Gender</th>
      <th>View Position</th>
      <th>OriginalImage[Width</th>
      <th>Height]</th>
      <th>OriginalImagePixelSpacing[x</th>
      <th>y]</th>
      <th>Unnamed: 11</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53382</th>
      <td>00013475_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>13475</td>
      <td>55</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>0.168</td>
      <td>NaN</td>
      <td>/data/images_006/images/00013475_000.png</td>
    </tr>
    <tr>
      <th>10584</th>
      <td>00002740_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>2740</td>
      <td>13</td>
      <td>M</td>
      <td>PA</td>
      <td>2992</td>
      <td>2991</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>NaN</td>
      <td>/data/images_002/images/00002740_000.png</td>
    </tr>
    <tr>
      <th>58068</th>
      <td>00014364_004.png</td>
      <td>No Finding</td>
      <td>4</td>
      <td>14364</td>
      <td>27</td>
      <td>F</td>
      <td>AP</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>0.168</td>
      <td>NaN</td>
      <td>/data/images_007/images/00014364_004.png</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Here you may want to create some extra columns in your table with binary indicators of certain diseases 
## rather than working directly with the 'Finding Labels' column

all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels({}):{}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1:
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)  
```

    All Labels(15):['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image Index</th>
      <th>Finding Labels</th>
      <th>Follow-up #</th>
      <th>Patient ID</th>
      <th>Patient Age</th>
      <th>Patient Gender</th>
      <th>View Position</th>
      <th>OriginalImage[Width</th>
      <th>Height]</th>
      <th>OriginalImagePixelSpacing[x</th>
      <th>...</th>
      <th>Emphysema</th>
      <th>Fibrosis</th>
      <th>Hernia</th>
      <th>Infiltration</th>
      <th>Mass</th>
      <th>No Finding</th>
      <th>Nodule</th>
      <th>Pleural_Thickening</th>
      <th>Pneumonia</th>
      <th>Pneumothorax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31770</th>
      <td>00008297_011.png</td>
      <td>Infiltration</td>
      <td>11</td>
      <td>8297</td>
      <td>18</td>
      <td>F</td>
      <td>AP</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.171</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>69184</th>
      <td>00017070_007.png</td>
      <td>No Finding</td>
      <td>7</td>
      <td>17070</td>
      <td>57</td>
      <td>M</td>
      <td>PA</td>
      <td>2992</td>
      <td>2991</td>
      <td>0.143</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>48287</th>
      <td>00012238_013.png</td>
      <td>No Finding</td>
      <td>13</td>
      <td>12238</td>
      <td>64</td>
      <td>M</td>
      <td>PA</td>
      <td>2992</td>
      <td>2991</td>
      <td>0.143</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 28 columns</p>
</div>




```python
all_xray_df['Pneumonia_class'] = ['Pneumonia' if x==1 else 'No Pneumonia' for x in all_xray_df['Pneumonia']]
all_xray_df.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image Index</th>
      <th>Finding Labels</th>
      <th>Follow-up #</th>
      <th>Patient ID</th>
      <th>Patient Age</th>
      <th>Patient Gender</th>
      <th>View Position</th>
      <th>OriginalImage[Width</th>
      <th>Height]</th>
      <th>OriginalImagePixelSpacing[x</th>
      <th>...</th>
      <th>Fibrosis</th>
      <th>Hernia</th>
      <th>Infiltration</th>
      <th>Mass</th>
      <th>No Finding</th>
      <th>Nodule</th>
      <th>Pleural_Thickening</th>
      <th>Pneumonia</th>
      <th>Pneumothorax</th>
      <th>Pneumonia_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1929</th>
      <td>00000499_010.png</td>
      <td>Nodule</td>
      <td>10</td>
      <td>499</td>
      <td>34</td>
      <td>F</td>
      <td>PA</td>
      <td>2992</td>
      <td>2991</td>
      <td>0.143</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No Pneumonia</td>
    </tr>
    <tr>
      <th>63894</th>
      <td>00015770_011.png</td>
      <td>Cardiomegaly</td>
      <td>11</td>
      <td>15770</td>
      <td>66</td>
      <td>F</td>
      <td>AP</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No Pneumonia</td>
    </tr>
    <tr>
      <th>59179</th>
      <td>00014639_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>14639</td>
      <td>54</td>
      <td>F</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No Pneumonia</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 29 columns</p>
</div>




```python
## Here we can create a new column called 'pneumonia_class' that will allow us to look at 
## images with or without pneumonia for binary classification


```

## Create your training and testing data:


```python
#def create_splits(vargs):
    
    ## Either build your own or use a built-in library to split your original dataframe into two sets 
    ## that can be used for training and testing your model
    ## It's important to consider here how balanced or imbalanced you want each of those sets to be
    ## for the presence of pneumonia
    
    # Todo
   
    #return train_data, val_data
```


```python
  train_df, valid_df = skl.train_test_split(all_xray_df, 
                                   test_size = 0.2, 
                                   stratify = all_xray_df['Pneumonia'])
```


```python
train_df['Pneumonia'].sum()/len(train_df)
```




    0.012765340706386016




```python
valid_df['Pneumonia'].sum()/len(valid_df)
```




    0.012754191937210132




```python
p_inds = train_df[train_df.Pneumonia==1].index.tolist()
np_inds = train_df[train_df.Pneumonia==0].index.tolist()

np_sample = sample(np_inds,len(p_inds))
train_df = train_df.loc[p_inds + np_sample]
```


```python
train_df['Pneumonia'].sum()/len(train_df)
```




    0.5




```python
p_inds = valid_df[valid_df.Pneumonia==1].index.tolist()
np_inds = valid_df[valid_df.Pneumonia==0].index.tolist()

np_sample = sample(np_inds,4*len(p_inds))
valid_df = valid_df.loc[p_inds + np_sample]
```


```python
valid_df['Pneumonia'].sum()/len(valid_df)
```




    0.2



# Now we can begin our model-building & training

#### First suggestion: perform some image augmentation on your data


```python
def my_image_augmentation(vargs):
    
    ## recommendation here to implement a package like Keras' ImageDataGenerator
    ## with some of the built-in augmentations 
    
    ## keep an eye out for types of augmentation that are or are not appropriate for medical imaging data
    ## Also keep in mind what sort of augmentation is or is not appropriate for testing vs validation data
    
    ## STAND-OUT SUGGESTION: implement some of your own custom augmentation that's *not*
    ## built into something like a Keras package
    
    # Todo
    
    return my_idg


def make_train_gen(vargs):
    
    ## Create the actual generators using the output of my_image_augmentation for your training data
    ## Suggestion here to use the flow_from_dataframe library, e.g.:
    
#     train_gen = my_train_idg.flow_from_dataframe(dataframe=train_df, 
#                                          directory=None, 
#                                          x_col = ,
#                                          y_col = ,
#                                          class_mode = 'binary',
#                                          target_size = , 
#                                          batch_size = 
#                                          )
     # Todo

    return train_gen


def make_val_gen(vargs):
    
#     val_gen = my_val_idg.flow_from_dataframe(dataframe = val_data, 
#                                              directory=None, 
#                                              x_col = ,
#                                              y_col = ',
#                                              class_mode = 'binary',
#                                              target_size = , 
#                                              batch_size = ) 
    
    # Todo
    return val_gen
```


```python
IMG_SIZE = (224, 224)
```


```python
my_idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=20,
                              shear_range = 0.1,
                              zoom_range=0.1) ## Here I'm adding a lot more zoom 

train_gen = my_idg.flow_from_dataframe(dataframe=train_df, 
                                         directory=None, 
                                         x_col = 'path',
                                         y_col = 'Pneumonia_class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 20
                                         )

val_gen = my_idg.flow_from_dataframe(dataframe=valid_df, 
                                         directory=None, 
                                         x_col = 'path',
                                         y_col = 'Pneumonia_class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 6)
```

    Found 2290 validated image filenames belonging to 2 classes.
    Found 1430 validated image filenames belonging to 2 classes.



```python
## May want to pull a single large batch of random validation data for testing after each epoch:
valX, valY = val_gen.next()
```


```python
## May want to look at some examples of our augmented training data. 
## This is helpful for understanding the extent to which data is being manipulated prior to training, 
## and can be compared with how the raw data look prior to augmentation

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        c_ax.set_title('Pneumonia')
    else:
        c_ax.set_title('No Pneumonia')
    c_ax.axis('off')
```


![png](output_22_0.png)


## Build your model: 

Recommendation here to use a pre-trained network downloaded from Keras for fine-tuning


```python
#def load_pretrained_model(vargs):
    
    #model = VGG16(include_top=True, weights='imagenet')
    #transfer_layer = model.get_layer('lay_of_interest')
    #vgg_model= Model(inputs = model.input, outputs = transfer_layer.output)
    
   #TODO
    
    #return vgg_model

```


```python
model = VGG16(include_top=True, weights='imagenet')
transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
for layer in vgg_model.layers[0:17]:
    layer.trainable = False
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    553467904/553467096 [==============================] - 8s 0us/step



```python
#def build_my_model(vargs):
    
    #my_model = Sequential()
    # ....add your pre-trained model, and then whatever additional layers you think you might
    # want for fine-tuning (Flatteen, Dense, Dropout, etc.)
    
    # if you want to compile your model within this function, consider which layers of your pre-trained model, 
    # you want to freeze before you compile 
    
    # also make sure you set your optimizer, loss function, and metrics to monitor
    
    # Todo
    
    #return my_model



## STAND-OUT Suggestion: choose another output layer besides just the last classification layer of your modele
## to output class activation maps to aid in clinical interpretation of your model's results
```


```python
my_model = Sequential()
my_model.add(vgg_model)
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(256, activation='relu'))
my_model.add(Dense(1, activation='sigmoid'))
    
```


```python
my_model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    model_1 (Model)              (None, 7, 7, 512)         14714688  
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 25088)             0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 25088)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              25691136  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               524800    
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 41,062,209
    Trainable params: 28,707,329
    Non-trainable params: 12,354,880
    _________________________________________________________________



```python
## Below is some helper code that will allow you to add checkpoints to your model,
## This will save the 'best' version of your model by comparing it to previous epochs of training

## Note that you need to choose which metric to monitor for your model's 'best' performance if using this code. 
## The 'patience' parameter is set to 10, meaning that your model will train for ten epochs without seeing
## improvement before quitting

# Todo

# weight_path="{}_my_model.best.hdf5".format('xray_class')

# checkpoint = ModelCheckpoint(weight_path, 
#                              monitor= CHOOSE_METRIC_TO_MONITOR_FOR_PERFORMANCE, 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode= CHOOSE_MIN_OR_MAX_FOR_YOUR_METRIC, 
#                              save_weights_only = True)

# early = EarlyStopping(monitor= SAME_AS_METRIC_CHOSEN_ABOVE, 
#                       mode= CHOOSE_MIN_OR_MAX_FOR_YOUR_METRIC, 
#                       patience=10)

# callbacks_list = [checkpoint, early]
```


```python
weight_path="{}_my_model.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, 
                             monitor= "val_loss", 
                             verbose=1, 
                             save_best_only=True, 
                             mode= "min", 
                             save_weights_only = True)

early = EarlyStopping(monitor= "val_loss", 
                      mode= "min", 
                      patience=10)

callbacks_list = [checkpoint, early]
```


```python
weight_path
```




    'xray_class_my_model.best.hdf5'




```python
optimizer = Adam(lr=1e-4)
loss = 'binary_crossentropy'
metrics = ['binary_accuracy']
```


```python
my_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
```

### Start training! 


```python
## train your model

# Todo

history = my_model.fit_generator(train_gen, 
                          validation_data = (valX, valY), 
                          epochs = 10, 
                          callbacks = callbacks_list)
```

    Epoch 1/10
    115/115 [==============================] - 2940s 26s/step - loss: 0.7778 - binary_accuracy: 0.5162 - val_loss: 0.8612 - val_binary_accuracy: 0.1667
    
    Epoch 00001: val_loss improved from inf to 0.86118, saving model to xray_class_my_model.best.hdf5
    Epoch 2/10
    115/115 [==============================] - 3012s 26s/step - loss: 0.7062 - binary_accuracy: 0.5367 - val_loss: 0.6286 - val_binary_accuracy: 0.8333
    
    Epoch 00002: val_loss improved from 0.86118 to 0.62860, saving model to xray_class_my_model.best.hdf5
    Epoch 3/10
    115/115 [==============================] - 2706s 24s/step - loss: 0.6956 - binary_accuracy: 0.5624 - val_loss: 0.6207 - val_binary_accuracy: 0.8333
    
    Epoch 00003: val_loss improved from 0.62860 to 0.62067, saving model to xray_class_my_model.best.hdf5
    Epoch 4/10
    115/115 [==============================] - 1505s 13s/step - loss: 0.6899 - binary_accuracy: 0.5537 - val_loss: 0.6158 - val_binary_accuracy: 0.8333
    
    Epoch 00004: val_loss improved from 0.62067 to 0.61580, saving model to xray_class_my_model.best.hdf5
    Epoch 5/10
    115/115 [==============================] - 1490s 13s/step - loss: 0.6814 - binary_accuracy: 0.5677 - val_loss: 0.6302 - val_binary_accuracy: 0.8333
    
    Epoch 00005: val_loss did not improve from 0.61580
    Epoch 6/10
    115/115 [==============================] - 1481s 13s/step - loss: 0.6822 - binary_accuracy: 0.5633 - val_loss: 0.7023 - val_binary_accuracy: 0.5000
    
    Epoch 00006: val_loss did not improve from 0.61580
    Epoch 7/10
    115/115 [==============================] - 1490s 13s/step - loss: 0.6782 - binary_accuracy: 0.5812 - val_loss: 0.6005 - val_binary_accuracy: 0.8333
    
    Epoch 00007: val_loss improved from 0.61580 to 0.60047, saving model to xray_class_my_model.best.hdf5
    Epoch 8/10
    115/115 [==============================] - 1485s 13s/step - loss: 0.6707 - binary_accuracy: 0.5852 - val_loss: 0.5731 - val_binary_accuracy: 0.8333
    
    Epoch 00008: val_loss improved from 0.60047 to 0.57315, saving model to xray_class_my_model.best.hdf5
    Epoch 9/10
    115/115 [==============================] - 1483s 13s/step - loss: 0.6664 - binary_accuracy: 0.5930 - val_loss: 0.5946 - val_binary_accuracy: 0.8333
    
    Epoch 00009: val_loss did not improve from 0.57315
    Epoch 10/10
    115/115 [==============================] - 1503s 13s/step - loss: 0.6659 - binary_accuracy: 0.5939 - val_loss: 0.5765 - val_binary_accuracy: 0.8333
    
    Epoch 00010: val_loss did not improve from 0.57315


##### After training for some time, look at the performance of your model by plotting some performance statistics:

Note, these figures will come in handy for your FDA documentation later in the project


```python
## After training, make some predictions to assess your model's overall performance
## Note that detecting pneumonia is hard even for trained expert radiologists, 
## so there is no need to make the model perfect.
my_model.load_weights(weight_path)
pred_Y = my_model.predict(valX, batch_size = 32, verbose = True)
```

    6/6 [==============================] - 3s 576ms/step



```python
print(pred_Y)
```

    [[0.3082655 ]
     [0.44801426]
     [0.5992588 ]
     [0.39581922]
     [0.42957243]
     [0.59294266]]



```python
def plot_history(history):
    N = len(history.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["binary_accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
```


```python
plot_history(history)

# Todo
```


![png](output_40_0.png)



```python
def plot_auc(t_y, p_y):
    
    ## Hint: can use scikit-learn's built in functions here like roc_curve
    
    # Todo
    
    return

## what other performance statistics do you want to include here besides AUC? 


# def ... 
# Todo

# def ...
# Todo
    
#Also consider plotting the history of your model training:

def plot_history(history):
    
   
    # Todo
    return
```


```python
def plot_auc(t_y, p_y):
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    fpr, tpr, thresholds = roc_curve(t_y, p_y)
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % ('Pneumonia', auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    
```


```python
plot_auc(valY, pred_Y)
```


![png](output_43_0.png)



```python
def plot_pr(t_y, p_y):
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_ax.plot(recall, precision, label = '%s (AP Score:%0.2f)'  % ('Pneumonia', average_precision_score(t_y,p_y)))
    c_ax.legend()
    c_ax.set_xlabel('Recall')
    c_ax.set_ylabel('Precision')


def  calc_f1(prec,recall):
    return 2*(prec*recall)/(prec+recall)

```


```python
plot_pr(valY, pred_Y)
```


![png](output_45_0.png)



```python
# function to calculate the F1 score
def  calc_f1(prec,recall):
    return 2*(prec*recall)/(prec+recall)
```


```python

plt.rcParams.update({'figure.max_open_warning': 0})
```


```python
def plot_f1s(valY,pred_y):
        
    precision, recall, thresholds = precision_recall_curve(valY.astype(int), pred_Y)
    f1_scores = []
    for i in thresholds:
        f1 = f1_score(valY.astype(int), binarize(pred_Y,i))
        f1_scores.append(f1)
    fig, c_ax = plt.subplots(1,1, figsize = (10, 10))
    c_ax.plot(thresholds, f1_scores, label = 'F1 Score')
    c_ax.legend()
    c_ax.set_xlabel('Threshold')
    c_ax.set_ylabel('F1 Score')
```


```python
plot_f1s(valY, pred_Y)
```


![png](output_49_0.png)



```python
precision, recall, thresholds = precision_recall_curve(valY.astype(int), pred_Y)
```


```python
precision_value = 0.7
idx = (np.abs(precision - precision_value)).argmin() 
print('Precision is: '+ str(precision[idx]))
print('Recall is: '+ str(recall[idx]))
print('Threshold is: '+ str(thresholds[idx]))
print('F1 Score is: ' + str(calc_f1(precision[idx],recall[idx])))
```

    Precision is: 1.0
    Recall is: 1.0
    Threshold is: 0.5992588
    F1 Score is: 1.0



```python
recall_value = 0.75
idx = (np.abs(recall - recall_value)).argmin() 
print('Precision is: '+ str(precision[idx]))
print('Recall is: '+ str(recall[idx]))
print('Threshold is: '+ str(thresholds[idx]))
print('F1 Score is: ' + str(calc_f1(precision[idx],recall[idx])))
```

    Precision is: 1.0
    Recall is: 1.0
    Threshold is: 0.5992588
    F1 Score is: 1.0



```python
probs = pred_Y
t1 = (probs > 0.39 )
t2 = (probs > 0.37)
```


```python
x1 = np.asarray(t1).astype(int)
x2 = np.asarray(t2).astype(int)
```


```python
x1
```




    array([[0],
           [1],
           [1],
           [1],
           [1],
           [1]])




```python
x1.ndim
```




    2




```python
x1=x1.flatten()
x2=x2.flatten()
x1=pd.Series(x1)
x2=pd.Series(x2)
```


```python
compare_t1 = (x1 == val_Y)
compare_t2 = (x2 == pred_Y)
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-58-b4aed7edb229> in <module>
    ----> 1 compare_t1 = (x1 == pred_Y)
          2 compare_t2 = (x2 == pred_Y)


    /opt/conda/lib/python3.7/site-packages/pandas/core/ops/common.py in new_method(self, other)
         62         other = item_from_zerodim(other)
         63 
    ---> 64         return method(self, other)
         65 
         66     return new_method


    /opt/conda/lib/python3.7/site-packages/pandas/core/ops/__init__.py in wrapper(self, other)
        526         res_values = comparison_op(lvalues, rvalues, op)
        527 
    --> 528         return _construct_result(self, res_values, index=self.index, name=res_name)
        529 
        530     wrapper.__name__ = op_name


    /opt/conda/lib/python3.7/site-packages/pandas/core/ops/__init__.py in _construct_result(left, result, index, name)
        473     # We do not pass dtype to ensure that the Series constructor
        474     #  does inference in the case where `result` has object-dtype.
    --> 475     out = left._constructor(result, index=index)
        476     out = out.__finalize__(left)
        477 


    /opt/conda/lib/python3.7/site-packages/pandas/core/series.py in __init__(self, data, index, dtype, name, copy, fastpath)
        303                     data = data.copy()
        304             else:
    --> 305                 data = sanitize_array(data, index, dtype, copy, raise_cast_failure=True)
        306 
        307                 data = SingleBlockManager(data, index, fastpath=True)


    /opt/conda/lib/python3.7/site-packages/pandas/core/construction.py in sanitize_array(data, index, dtype, copy, raise_cast_failure)
        480     elif subarr.ndim > 1:
        481         if isinstance(data, np.ndarray):
    --> 482             raise Exception("Data must be 1-dimensional")
        483         else:
        484             subarr = com.asarray_tuplesafe(data, dtype=dtype)


    Exception: Data must be 1-dimensional



```python
compare_t1
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-59-8ed6dae73c3c> in <module>
    ----> 1 compare_t1
    

    NameError: name 'compare_t1' is not defined



```python
print('Accuracy at threshold 1: ' + str(len(compare_t1[compare_t1])/len(pred_Y)))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-60-dba889a749ac> in <module>
    ----> 1 print('Accuracy at threshold 1: ' + str(len(compare_t1[compare_t1])/len(pred_Y)))
    

    NameError: name 'compare_t1' is not defined



```python
print('Accuracy at threshold 2: ' + str(len(compare_t2[compare_t2])/len(pred_Y)))
```

Once you feel you are done training, you'll need to decide the proper classification threshold that optimizes your model's performance for a given metric (e.g. accuracy, F1, precision, etc.  You decide) 


```python
## Find the threshold that optimize your model's performance,
## and use that threshold to make binary classification. Make sure you take all your metrics into consideration.

# Todo
```


```python
## Let's look at some examples of true vs. predicted with our best model: 

# Todo

# fig, m_axs = plt.subplots(10, 10, figsize = (16, 16))
# i = 0
# for (c_x, c_y, c_ax) in zip(valX[0:100], testY[0:100], m_axs.flatten()):
#     c_ax.imshow(c_x[:,:,0], cmap = 'bone')
#     if c_y == 1: 
#         if pred_Y[i] > YOUR_THRESHOLD:
#             c_ax.set_title('1, 1')
#         else:
#             c_ax.set_title('1, 0')
#     else:
#         if pred_Y[i] > YOUR_THRESHOLD: 
#             c_ax.set_title('0, 1')
#         else:
#             c_ax.set_title('0, 0')
#     c_ax.axis('off')
#     i=i+1
```


```python
fig, m_axs = plt.subplots(10, 10, figsize = (16, 16))
i = 0
for (c_x, c_y, c_ax) in zip(valX[0:100], valY[0:100], m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        if pred_Y[i] > 0.5:
            c_ax.set_title('1, 1')
        else:
             c_ax.set_title('1, 0')
    else:
        if pred_Y[i] > 0.5: 
            c_ax.set_title('0, 1')
        else:
             c_ax.set_title('0, 0')
    c_ax.axis('off')
    i=i+1
```


```python

```


```python
## Just save model architecture to a .json:

model_json = my_model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)
```
