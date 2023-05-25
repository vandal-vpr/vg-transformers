
# Learning Sequential Descriptors for Sequence-based Visual Place Recognition

<p align="center">
  <img src="./assets/fams.png">
    <br/><em>Taxonomy of sequential descriptor methods.</em>
</p>

This is the official repository for the paper "[Learning Sequential Descriptors for Sequence-based Visual Place Recognition](https://arxiv.org/abs/2207.03868)".
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
python main_scripts/msls/1_reformat_mapillary.py  original_MSLS/folder/path destination/folder/path
python main_scripts/msls/2_reformat_testset_msls.py  reformatted/MSLS/path
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

We provide the 2 pre-processed versions that we used in our experiments:
* Fixed-Space sampling, keeping one frame every 2 meters: [link](https://drive.google.com/file/d/17QGrkRN9tZ88eLld4ptY6hMZOEhkvjoS/view?usp=sharing)
* Fixed-Time sampling, keeping one frame every 3.6 seconds: [link](https://drive.google.com/file/d/1QlfV7fliuTl9AvHeuyjMLr4kV6BAJOfn/view?usp=sharing)

The first one is more consistent with the MSLS setup. For the second one, the choice of the 3.6 seconds threshold was made to keep a comparable number of images with the first version.

Alternatively, you can download the full,raw version of the dataset from the official website it and preprocess the dataset use the following commands:

``` bash 
python main_scripts/robotcar/1_downloader.py
python main_scripts/robotcar/2_untar.py
python main_scripts/robotcar/3_dataset_builder_all.py
python main_scripts/robotcar/4_reduce_density.py
python main_scripts/robotcar/5_format_tree.py
```
</details>

## Model zoo

We are currently exploring hosting options, so this is a partial list of models. More models will be added soon!!

<details>
    <summary><b>Pretrained models with SeqVLAD and different backbones</b></summary></br>
    Pretained networks employing different backbones.</br></br>
	<table>
		<tr>
			<th rowspan=2>Model</th>
			<th colspan="3">Training on MSLS, seq len 5</th>
	 	</tr>
	 	<tr>
	   		<td>MSLS (R@1)</td>
	   		<td>Download</td>
	 	</tr>
		<tr>
			<td>CCT384 + SeqVLAD</td>
			<td>89.6</td>
			<td><a href="https://drive.google.com/file/d/16n6CL2t-asQ_tf8x4ZyJT_Y4UQQedsxh/view?usp=sharing">[Link]</a></td>
	 	</tr>
	</table>
</details>

## Run Experiments
Once the datasets are ready, we can proceed running the experiments with the architecture of choice.

**NB**: to build MSLS sequences, some heavy pre-processing to build data structures is needed. The dataset class will automatically cache this,
so to compute them only the first time. Therefore the first experiment that you ever launch will take 2-3 hours to build this structures which will
be saved in a `cache` directory, and following experiments will then start quickly. Note that this procedure caches everything with relative paths,
therefore if you want to run experiments on multiple machines you can simply copy the `cache` directory.
Finally, note that this data structures must be computed for each sequence length, so potentially in `cache` you will have a file for each sequence_length
that you want to experiment with.

**TODO one for each family of methods (also consider put an example with RobotCar)**

Example with CCT-384 + SeqVLAD on MSLS:
``` bash 
python main_scripts/main_train.py \
	--dataset_path <MSLS path>
	--img_shape 384 384 \
	--arch cct384 --aggregation seqvlad \
	--trunc_te 8 --freeze_te 1 \
	--train_batch_size 4 --nNeg 5 --seq_length 5 \
	--optim adam --lr 0.00001
```

Example with TimeSformer:
``` bash 
python main_scripts/main_train.py \
	--dataset_path <MSLS path>
	--img_shape 224 224 \
	--arch timesformer --aggregation _ \
	--train_batch_size 4 --nNeg 5 --seq_length 5 \
	--optim adam --lr 0.00001
```

Example with ResNet-18 + GeM + CAT :
``` bash
python main_scripts/main_train.py \
	--dataset_path <MSLS path>
	--img_shape 480 640 \
	--arch r18l3 --pooling gem --aggregation cat \
	--train_batch_size 4 --nNeg 5 --seq_length 5 \
	--optim adam --lr 0.00001
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

 
#### Other related Projects

[Deep Visual Geo-Localization Benchmark](https://github.com/gmberton/benchmarking_vg)

#### Resources used in this work
[Official SeqNet implementation](https://github.com/oravus/seqNet)

[Official SeqMatchNet implementation](https://github.com/oravus/SeqMatchNet)

[CCT repository](https://github.com/SHI-Labs/Compact-Transformers)


## Cite
Here is the bibtex to cite our paper
```
@article{Mereu_2022_seqvlad,
  author={Mereu, Riccardo and Trivigno, Gabriele and Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
  journal={IEEE Robotics and Automation Letters},
  title={Learning Sequential Descriptors for Sequence-Based Visual Place Recognition}, 
  year={2022},
  volume={7},
  number={4},
  pages={10383-10390},
  doi={10.1109/LRA.2022.3194310}
}
```
