# FaceID-GAN
this is a re-implementation of CVPR2018 paper "FaceID-GAN" using Pytorch

note: I don't finish this work because model is so big that my compute doesn't support training, but I haved done most of work such as training a P net according to paper and building model arctiture. Maybe you need to optimize my code, because I don't test it.

# dependencies
* Pytorch >=0.4
* python3
* pip install face-alignment

# get pre-trained P net
* firstly to get 68 key points of one face, using this [great work](https://github.com/1adrianb/face-alignment)
* using [open source code](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/HPEN/main.htm) to get 3DMM parameters for training P net
* change "./train_p/coonfig.py" and "./train_p/vector_loader.py" to adapt to your environment.
* run "./train_p/train_pnet.py", and p outputs 235-dims vector whose formulation is [Pose_Para;Shape_Para;Exp_Para]

notes that I have trained p net, you can download from [here](https://pan.baidu.com/s/17hJMV7jJVJSruHWkfBr1lA). Code is "1yvm".

# acknowledgement
* thanks for first author of FaceID-GAN to reply my email and untie my confusion.
* thanks for the wonderful open source works to develop this project,such as [BEGAN](https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/BEGAN.py),face_alignment.
