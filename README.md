# LungSegmentation
Segmentation of the heart and lungs of the JSRT - Chest Lung Nodules and Non-Nodules images data set using UNet, R2U-Net and DCAN

<img src="https://github.com/topinfrassi01/LungSegmentation/raw/master/imgs/summary.PNG"/>

## Dataset descriptions

- The x-ray database is provided by the Japanese Society of Radiological Technology (JSRT) in cooperation with the Japanese Radiological Society (JRS) and is available at this address : http://db.jsrt.or.jp/eng.php.
- The masks database is provided by the Image Sciences Institue with regards to the x-ray database previously described.

More exhaustive documentation can be found in the documentation folder inside the Git.

## Data preprocessing

In order to prepare the data, ImageJ macros have been created to transform the x-ray images and the masks to a common PNG format.

Afterwards, run the combine_lungs.py script to combine the masks of both lungs in a single image, you can then run preprocess_images.py to create the .npy archives that will be used in the training.

## Network training

Three architectures were tested : 
- (U-Net)[https://arxiv.org/pdf/1505.04597.pdf]
- (Recurrent Residual U-Net)[https://arxiv.org/abs/1802.06955]
- (Deep Contour-Aware Network)[https://arxiv.org/abs/1604.02677]

In the models.py file, there are implementation using Tensorflow's Keras of those three networks.
There are also run_*.py files used to train each of the networks.

## Results

<img src="https://github.com/topinfrassi01/LungSegmentation/raw/master/imgs/results.PNG"/>

Appart from these results, we've noticed that the bottom part of the right lung is usually harder to segment because it's generally harder to see.
In addition to this, the conditions that determine if a pixel belongs to the heart category make it hard to qualitatively evaluate the results.