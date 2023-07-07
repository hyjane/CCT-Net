Simple Image Preprocessing:Run resize.py to unify the image size to (448,448) and save it in npy format. You need to manually adjust the directory.Store the processed image according to the path in 1.

1. The datas used to store the input in datas
storage folder
--datas
----ISIC2016
------Train
--------Image
--------Label
------Test
--------Image
--------Label
------Validation
--------Image
--------Label
----ISIC2017
------Train
--------Image
--------Label
------Test
--------Image
--------Label
------Validation
--------Image
--------Label
----ISIC2018
------Train
--------Image
--------Label
------Test
--------Image
--------Label
------Validation
--------Image
--------Label
----PH2
------Train
--------Image
--------Label
------Test
--------Image
--------Label
------Validation
--------Image
--------Label

2. Example of the method of training and testing the model: When using ISIC2016 for training and testing, you need to put the ISIC2016 Train, Test, and Validation in the datas path. You also need need to make a new folder to save the output.
--datas
----Train
----Test
----Validation
----Output

3. Training model: run Train.py

4. Testing model：run infer.py
