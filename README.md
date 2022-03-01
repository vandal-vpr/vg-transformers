
# Learning Sequential Descriptors for Sequence-based Visual Place Recognition

<p align="center">
  <img src="./assets/fams.png">
    <br/><em>Taxonomy of sequential descriptor methods.</em>
</p>

This is the official repository for the paper "Learning Sequential Descriptors for Sequence-based Visual Place Recognition".
It can be used to reproduce results from the paper and experiment with a wide range of sequential descriptor methods for Visual Place Recognition. 



## Install Locally
Create your local environment and then install the required packages using:
``` bash
pip install -r pip_requirements.txt
# to install the official TimeSformer package
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer  
python setup.py build develop
```
## Datasets
The experiments in the paper use two main datasets Mapillary Street Level Sequence (MSLS) and Oxford RobotCar.
<details>
  <summary><b>MSLS</b></summary></br>
Download the dataset from <a href="https://github.com/mapillary/mapillary_sls">here</a> and then reformat the file using:

``` bash
python main_scripts/msls/1_reformat_mapillary.py
python main_scripts/msls/2_reformat_testset_msls.py
```

Moreover, to reduce the time needed to compute the sequences of a certain sequence length at each run it is possibile to create a cache file for the MSLS training set by using:

``` bash
python main_scripts/3_cache_dataset.py --seq <put your seq_len> --dataset_path <path to MSLS dataset> --save_to <output path>
```

</details>

<details>
  <summary><b>Oxford RobotCar</b></summary></br>
In our experiments, we used the following laps of Oxford RobotCar as train/validation/test sets:

*  Train set: 
	* queries:   lap 2014-12-17-18-18-43 (winter night, rain);
	* database: lap  2014-12-16-09-14-09 (winter day, sun);
*  Validation set:
	* queries:  lap 2015-02-03-08-45-10 (winter day, snow);
	* database: lap 2015-11-13-10-28-08 (fall day, overcast).
*  Test set :
	* queries: lap  2014-12-16-18-44-24 (winter night); 
	* database: lap 2014-11-18-13-20-12 (fall day).

To download and preprocess the dataset use the following commands:

``` bash 
python main_scripts/robotcar/1_downloader.py
python main_scripts/robotcar/2_untar.py
python main_scripts/robotcar/3_dataset_builder_all.py
python main_scripts/robotcar/4_reduce_density.py
python main_scripts/robotcar/5_format_tree.py
```
</details>

**TODO consider to put all steps in a single script if they do not require particular arguments** 

## Run Experiments
Once the datasets are ready, we can proceed running the experiments with the architecture of choice.
**TODO one for each family of methods (also consider put an example with RobotCar)**

Example with CCT-384 + SeqVLAD on MSLS:
``` bash 
python main_scripts/main_train.py \
	--dataset_path <MSLS path>
	--img_shape 384 384 \
	--arch cct384 --aggregation seqvlad \
	--trunc_te 8 --freeze_te 1 \
	--train_batch_size 4 --nNeg 5 --seq_length 5 \
	--optim adam --lr 0.0001
```

Example with TimeSformer:
``` bash 
python main_scripts/main_train.py \
	--dataset_path <MSLS path>
	--img_shape 224 224 \
	--arch timesformer --aggregation none \
	--train_batch_size 4 --nNeg 5 --seq_length 5 \
	--optim adam --lr 0.0001
```

Example with ResNet-18 + GeM + CAT :
``` bash
python main_scripts/main_train.py \
	--dataset_path <MSLS path>
	--img_shape 480 640 \
	--arch r18l3 --pooling gem --aggregation cat \
	--train_batch_size 4 --nNeg 5 --seq_length 5 \
	--optim adam --lr 0.0001
```

### Add PCA
To add the PCA to SeqVLAD or CAT models use:
 
``` bash 
python main_scripts/evaluation.py \
	--pca_outdim <descr. dim.> \
	--resume <path trained model w/o PCA> 
```
where the parameter `--pca_outdim` determines the final descriptor dimensionality (in our test we used 4096)

### Evaluate trained models 
It is possible to evaluate the trained models using:
``` bash 
python main_scripts/evaluation.py \
	--resume <path trained model>
```

 
#### Other realesed Projects

[Deep Visual Geo-Localization Benchmark](https://github.com/gmberton/benchmarking_vg)

#### Resources used in this work
[Official SeqNet implementation](https://github.com/oravus/seqNet)

[Official SeqMatchNet implementation](https://github.com/oravus/SeqMatchNet)

[CCT repository](https://github.com/SHI-Labs/Compact-Transformers)
