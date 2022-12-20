# MPILNet: Multi-level Patch Integrated Learning Network for Non-homogeneous Haze Removal
## Abstract
##### We observe that convolutional neural network (CNN) perform impressive de-hazing performance on large-scale homogeneous hazy datasets. To improve the generalization of model, we need to train the de-hazing model on large-scale non-homogeneous hazy dataset. However, such dataset is not easy to obtain. We leverage trained VGG to perform transfer learning on small-scale hazy dataset. Unlike increasing the depth of network to improve the performance of de-hazing, to obtain a complex balance between image details and overall semantics, we aggregate features of multiple image patches to obtain better performance on hierarchical architecture. We propose an integrated learning decoder, consists of internal integrated learning module and cross-level integrated learning module, to progressively restore the haze-free image in coarse-to-fine way. The internal integrated learning module decodes by gradually incorporating the deep and shallow features of the same patch. Cross-level integrated learning module infuses decoded features at current level into the next level.Our method does not perform best on all evaluation metrics in these four datasets, however, the effect of haze removal in the real hazy image is significantly better than that of other methods, showing excellent generalization. Our source code will be available on the GitHub: https://github.com/Ruini94/MPILNet
### 
***
## Experiments
### Requirements
* pytorch_msssim

### Dataset Preparation
#### Please download Rain1200,RainCityscapes and RainDS：
[RESIDE-6K](https://drive.google.com/drive/folders/10cP6Z-n2G0006_ppW1WxkQpNKg3mSfnj)  
[HazeCityscapes](https://www.cityscapes-dataset.com/downloads/)  
[Nitre19](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/files/Dense_Haze_NTIRE19.zip)
[Nitre20](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip)
[Nitre21](https://competitions.codalab.org/competitions/28032#participate)
#### please put datasets in
> data_path
>> trainA  
>> trainB  
>> testA  
>> testB
***
## Usage
### train
#### `python main.py -name RESIDE-6K -root /data/users/haze_dataset/RESIDE-6K`
### test
#### `python test_metrics.py`
