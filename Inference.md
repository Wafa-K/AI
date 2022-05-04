```python
import numpy as np
import pandas as pd
import pydicom
%matplotlib inline
import matplotlib.pyplot as plt
import keras 
import skimage
import glob
from keras.models import model_from_json
from skimage.transform import resize
from PIL import Image
```


```python
# This function reads in a .dcm file, checks the important fields for our device, and returns a numpy array
# of just the imaging data
def check_dicom(filename): 
    # todo
    
#     print('Load file {} ...'.format(filename))
#     ds = pydicom.dcmread(filename)       
#     img = ds.pixel_array
    return img
    
    
# This function takes the numpy array output by check_dicom and 
# runs the appropriate pre-processing needed for our model input
def preprocess_image(img,img_mean,img_std,img_size): 
    # todo
    
    return proc_img

# This function loads in our trained model w/ weights and compiles it 
def load_model(model_path, weight_path):
    # todo
    
    
    return model

# This function uses our device's threshold parameters to predict whether or not
# the image shows the presence of pneumonia using our trained model
def predict_image(model, img, thresh): 
    # todo    
    
    return prediction 
```


```python
mydicoms = glob.glob("*.dcm")
```


```python
ds = pydicom.dcmread(mydicoms[1])
```


```python
ds
```




    (0008, 0016) SOP Class UID                       UI: Secondary Capture Image Storage
    (0008, 0018) SOP Instance UID                    UI: 1.3.6.1.4.1.11129.5.5.110503645592756492463169821050252582267888
    (0008, 0060) Modality                            CS: 'DX'
    (0008, 1030) Study Description                   LO: 'No Finding'
    (0010, 0020) Patient ID                          LO: '2'
    (0010, 0040) Patient's Sex                       CS: 'M'
    (0010, 1010) Patient's Age                       AS: '81'
    (0018, 0015) Body Part Examined                  CS: 'RIBCAGE'
    (0018, 5100) Patient Position                    CS: 'PA'
    (0020, 000d) Study Instance UID                  UI: 1.3.6.1.4.1.11129.5.5.112507010803284478207522016832191866964708
    (0020, 000e) Series Instance UID                 UI: 1.3.6.1.4.1.11129.5.5.112630850362182468372440828755218293352329
    (0028, 0002) Samples per Pixel                   US: 1
    (0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
    (0028, 0010) Rows                                US: 1024
    (0028, 0011) Columns                             US: 1024
    (0028, 0100) Bits Allocated                      US: 8
    (0028, 0101) Bits Stored                         US: 8
    (0028, 0102) High Bit                            US: 7
    (0028, 0103) Pixel Representation                US: 0
    (7fe0, 0010) Pixel Data                          OW: Array of 1048576 elements




```python
#Modality 
#Study Description
#Patient ID
#Patient's Sex
#Patient's Age
#Body Part Examined
#Patient Position
#Rows
#Columns
```


```python
ds.BodyPartExamined
```




    'CHEST'




```python
ds.Modality
```




    'CT'




```python
Ta = pd.read_csv('sample_labels.csv')
Ta
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
      <th>OriginalImageWidth</th>
      <th>OriginalImageHeight</th>
      <th>OriginalImagePixelSpacing_x</th>
      <th>OriginalImagePixelSpacing_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000013_005.png</td>
      <td>Emphysema|Infiltration|Pleural_Thickening|Pneu...</td>
      <td>5</td>
      <td>13</td>
      <td>060Y</td>
      <td>M</td>
      <td>AP</td>
      <td>3056</td>
      <td>2544</td>
      <td>0.139000</td>
      <td>0.139000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000013_026.png</td>
      <td>Cardiomegaly|Emphysema</td>
      <td>26</td>
      <td>13</td>
      <td>057Y</td>
      <td>M</td>
      <td>AP</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168000</td>
      <td>0.168000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000017_001.png</td>
      <td>No Finding</td>
      <td>1</td>
      <td>17</td>
      <td>077Y</td>
      <td>M</td>
      <td>AP</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168000</td>
      <td>0.168000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000030_001.png</td>
      <td>Atelectasis</td>
      <td>1</td>
      <td>30</td>
      <td>079Y</td>
      <td>M</td>
      <td>PA</td>
      <td>2992</td>
      <td>2991</td>
      <td>0.143000</td>
      <td>0.143000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000032_001.png</td>
      <td>Cardiomegaly|Edema|Effusion</td>
      <td>1</td>
      <td>32</td>
      <td>055Y</td>
      <td>F</td>
      <td>AP</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168000</td>
      <td>0.168000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5601</th>
      <td>00030712_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>30712</td>
      <td>058Y</td>
      <td>M</td>
      <td>PA</td>
      <td>2021</td>
      <td>2021</td>
      <td>0.194311</td>
      <td>0.194311</td>
    </tr>
    <tr>
      <th>5602</th>
      <td>00030786_005.png</td>
      <td>Cardiomegaly|Effusion|Emphysema</td>
      <td>5</td>
      <td>30786</td>
      <td>061Y</td>
      <td>F</td>
      <td>AP</td>
      <td>3056</td>
      <td>2544</td>
      <td>0.139000</td>
      <td>0.139000</td>
    </tr>
    <tr>
      <th>5603</th>
      <td>00030789_000.png</td>
      <td>Infiltration</td>
      <td>0</td>
      <td>30789</td>
      <td>052Y</td>
      <td>F</td>
      <td>PA</td>
      <td>2021</td>
      <td>2021</td>
      <td>0.194311</td>
      <td>0.194311</td>
    </tr>
    <tr>
      <th>5604</th>
      <td>00030792_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>30792</td>
      <td>010Y</td>
      <td>F</td>
      <td>PA</td>
      <td>1775</td>
      <td>1712</td>
      <td>0.194311</td>
      <td>0.194311</td>
    </tr>
    <tr>
      <th>5605</th>
      <td>00030797_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>30797</td>
      <td>024Y</td>
      <td>M</td>
      <td>PA</td>
      <td>2021</td>
      <td>2021</td>
      <td>0.194311</td>
      <td>0.194311</td>
    </tr>
  </tbody>
</table>
<p>5606 rows × 11 columns</p>
</div>




```python
test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']
```


```python
# This function reads in a .dcm file, checks the important fields for our device, and returns a numpy array
# of just the imaging data
def check_dicom(filename): 
    all_data = []
    
    for i in test_dicoms:   
        ds = pydicom.dcmread(i)
        fields = [ds.BodyPartExamined, ds.Modality, ds.PatientPosition]
        all_data.append(fields)
    
        print('Load file {} ...'.format(filename))
        ds = pydicom.dcmread(filename)  
    
    if ds.BodyPartExamined != 'CHEST' or ds.Modality != 'DX' or ds.PatientPosition not in ['PA', 'AP'] or int(ds.PatientAge) > 110:
        img = ds.pixel_array       
        return img 
    else:
        print("Unable to process this file")
        
       

    
```


```python
ds = pydicom.dcmread('test1.dcm')
```


```python
# This function takes the numpy array output by check_dicom and 
# runs the appropriate pre-processing needed for our model input
def preprocess_image(img,img_mean,img_std,img_size): 
    img = ds.pixel_array
    img_mean = np.mean(img)
    img_std = np.std(img)
    img_size = (224, 224)
    proc_img = (img - img_mean)/img_std
    proc_img =  resize(proc_img, img_size, anti_aliasing=True)
    return proc_img
```


```python
# This function loads in our trained model w/ weights and compiles it 
def load_model(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    
    prediction = model.predict([img], batch_size = 1, verbose = True)
    return model
```


```python
# This function uses our device's threshold parameters to predict whether or not
# the image shows the presence of pneumonia using our trained model
def predict_image(model, img, thresh): 
    result = model.predict(img) 
    predict=result[0]
    prediction='No pneumonia'
    if(predict>thresh):
        prediction='Pneumonia'
        
    return prediction 
```


```python

```


```python
img=check_dicom ('test2.dcm')
```

    Load file test2.dcm ...
    Load file test2.dcm ...
    Load file test2.dcm ...
    Load file test2.dcm ...
    Load file test2.dcm ...
    Load file test2.dcm ...
    Unable to process this file



```python
 test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path = "my_model.json"
weight_path = 'xray_class_my_model.best.hdf5'

IMG_SIZE=(1,224,224,3) # This might be different if you did not use vgg16
img_mean = ?104.74800395965576 # loads the mean image value they used during training preprocessing
img_std = ?66.22413614301003
# loads the std dev image value they used during training preprocessing

my_model = load_model(model_path, weight_path)
thresh = #loads the threshold they chose for model classification 

# use the .dcm files to test your prediction
for i in test_dicoms:
    
    img = np.array([])
    img = check_dicom(i)
    
    if img is None:
        continue
        
    img_proc = preprocess_image(img,img_mean,img_std,IMG_SIZE)
    pred = predict_image(my_model,img_proc,thresh)
    print(pred)   
```


      File "<ipython-input-36-2b335b6588c3>", line 7
        img_mean = ?104.74800395965576 # loads the mean image value they used during training preprocessing
                   ^
    SyntaxError: invalid syntax




```python
test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path = #path to saved model
weight_path = #path to saved best weights

IMG_SIZE=(1,224,224,3) # This might be different if you did not use vgg16
img_mean = # loads the mean image value they used during training preprocessing
img_std = # loads the std dev image value they used during training preprocessing

my_model = #loads model
thresh = #loads the threshold they chose for model classification 

# use the .dcm files to test your prediction
for i in test_dicoms:
    
    img = np.array([])
    img = check_dicom(i)
    
    if img is None:
        continue
        
    img_proc = preprocess_image(img,img_mean,img_std,IMG_SIZE)
    pred = predict_image(my_model,img_proc,thresh)
    print(pred)
```
