# FastStereoNet: A Fast Neural Architecture Search for Improving the Accuracy of Disparity Estimation on Resource-Constrained Platforms (IEEE Trans. SMC)
This is a Tensorflow implementation of the FastStereoNet framework.

## Requirments
- Python 3.7
- Tensorflow==1.14
- [Simanneal](https://github.com/perrygeo/simanneal)
- [Lahc](https://github.com/gunnstein/lahc)
- Sqlite3==0.8
- Numpy==1.16
- Scipy==1.2.1
## Training data
The [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) dataset has been used for training. This dataset consists of total of 200 scenes for training and of 200 scenes for testing. For more details, please check the KITTI website.
## Pre-Processing
For training and validation, locations from the ground truth disparity images are generated using the preprocessing scripts published in the paper code available [here](https://bitbucket.org/saakuraa/cvpr16_stereo_public/src/1a41996ef7dda999b43d249fd51442d0b2e9dd0f/preprocess/?at=master). This preprocessing script generates 3 binary files. Only the training and validation binary files are used.
## Running the code
1. Clone the code :
```
git clone https://github.com/alizoljodi/Final_ICML/
```
2. Run main.py :
```
python main.py --model_dir=A_DIRECTORY_TO_SAVE_RESULTS \
--data_root=PATH_DATASET \
--util_root=PATH_UTIL_ROOT_DIRECTORY
```
## Results
The results of the running process would be saved in model_dir. It contains 2 subfolders which are named HC and SA. The results of the LAHC algorithm would be saved in HC and SA results would be saved in SA.
each subfolder contains a *.db file named 'bests' for running summaries.
