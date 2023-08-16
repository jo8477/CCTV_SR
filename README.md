# CCTV-SR
Face detection and SR in cctv images

## Datasets

### Train、Val Dataset
The train and validation datasets use the Labeled Faces in the Wild (LFW) dataset.
Train dataset has 11910 images and Val dataset has 1323 images.
Download the datasets from [here][(https://pan.baidu.com/s/1xuFperu2WiYc5-_QXBemlA)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset), 
and then extract it into `data` directory. Finally run
```
python data_uti.py
```
Create training and Val data sets in LFW with an upscale factor of 4.

### Test Image Dataset
When you run the facedect.py file, it detects your face from the ./data/video/face.mp4 image and saves the detected image once every 10 frames.


### Train

If visdom is not installed, you must install it.

```
pip install visdom
```

### Test Image
```
python img_gen.py
```
The output high resolution images are on `results` directory.

### View result
```
python resultview.py
```

