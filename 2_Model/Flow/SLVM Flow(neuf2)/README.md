# SLVMFlow [This is a fast access version not the end]
([NeuFlow2 based](https://arxiv.org/abs/2408.10161))

## Inference
```
python infer.py
```

<img src="outputs/test_results2/000016_10.png" width="400" >

## Datasets

The datasets used to train and evaluate NeuFlow are as follows:

* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) 
* [DarkTraffic](https://drive.google.com/drive/folders/1tqOh4ZqGeK6BJd_aCvbMETwD9kaaZkGF?usp=drive_link)

By default the dataloader assumes the datasets are located in folder `datasets` and are organized as follows:

```
datasets
├── FlyingChairs_release
│   └── data
├── FlyingThings3D
│   ├── frames_cleanpass
│   ├── frames_finalpass
│   └── optical_flow
├── HD1K
│   ├── hd1k_challenge
│   ├── hd1k_flow_gt
│   ├── hd1k_flow_uncertainty
│   └── hd1k_input
├── KITTI @ Darktraffic
│   ├── testing
│   └── training
├── Sintel
│   ├── test
│   └── training
```

Symlink your dataset root to `datasets`:

```shell
ln -s $YOUR_DATASET_ROOT datasets
```

Convert all your images and flows to .npy format to speed up data loading. This script provides an example of converting FlyingThings cleanpass data.
```
python images_flows_to_npy.py
```

## Training

Simple training script:
```
python train.py \
--model SLVMFlow
--checkpoint_dir $YOUR_CHECKPOINT_DIR \
--stage things \
--val_dataset things sintel kitti \
--batch_size 32 \
--num_workers 4 \
--lr 1e-4 \
--val_freq 1000 \
--strict_resume
```