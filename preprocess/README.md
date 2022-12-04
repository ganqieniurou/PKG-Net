## Preparing

### 1. Objects detection preparing
To extract STCs, please install the [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmaction2) accordingly, then download the cascade RCNN [pretrained weights](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) (we use `cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth`) and place it in `pre_porocess/assets` folder.

### 2. Dataset preparing

Please download [UCSDPed2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html), [CUHKAvenue](http://101.32.75.151:8181/dataset/), [ShanghaiTech](http://101.32.75.151:8181/dataset/) dataset and place it into the `preprocess` directory of this project. The file structure should be similar as follows:

```python
./preprocess
└── ped2
│   ├── testing
│   │   └── frames
│   │       ├── Test001
│   │       ├── Test001_gt
│   │       ├── Test002
│   │       ├── Test002_gt
│   │       └── ...
│   └── training
│        └── frames
│            ├── Train001
│            ├── Train002
│            └── ...
└── avenue
│   ├── testing
│   │   └── frames
│   │       ├── 01
│   │       ├── 02
│   │       └── ...
│   └── training
│        └── frames
│            ├── 01
│            ├── 02
│            └── ...
└── shtech
    ├── testing
    │   └── frames
    │       ├── 01_0014
    │       ├── 01_0015
    │       └── ...
    └── training
         └── frames
             ├── 01_001
             ├── 01_002
             └── ...
```
>For the ShanghaiTech dataset, it only contains raw videos as the training data. One can use ffmpeg tools to extract the frames, such as `ffmpeg -i <video_name> -qscale:v 1 -qmin 1 <video_name/%04d.jpg>`. 
