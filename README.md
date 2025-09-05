# [A Biologically Inspired Separable Learning Vision Model for Real-time Traffic Object Perception in Dark](https://doi.org/10.1016/j.eswa.2025.129529)

[简体中文](README-zh-CN.md) | [English](README.md)
<br>

![framwk](figs/framwk.png)

## Dark-traffic benchmark
(object detection, instance segmentation, and optical flow estimation in the low-light traffic conditions)

images (~10k) and annotations (~100k) in [Google Drive](https://drive.google.com/drive/folders/1B8EzDn64bGBgyRCfppL_jhcOA3hIwnzi?usp=sharing)

The images are now available, and annotations will be released after the paper is processed by the journal/conference.

- 9.25 update: Available on _Expert Systems with Applications_, annotations is released.

### SLVM for optical flow
![flow](figs/f10.png)

## Re-produce
        python pip install -requirements.txt
 ## References
  - https://github.com/AlanLi1997/slim-neck-by-gsconv
  - https://github.com/AlanLi1997/rethinking-fpn
  - https://github.com/ultralytics/ultralytics
  - https://github.com/haofeixu/gmflow
  - https://github.com/neufieldrobotics/NeuFlow_v2
