《基于Keras的交通标志识别》实操备忘笔记

---

**前言**

　　本笔记基于腾讯云的一个免费课程，仅对实操过程做简单的记录，原课程使用CPU进行训练，相同环境下，官方实验手册已经描述的非常详尽，以下笔记是基于GPU进行训练的记录。（本人非专业外行菜鸟一枚，内容~~或有~~肯定有错漏）

---

传送门：

[在线课程](https://cloud.tencent.com/edu/learning/course-3423-60120)

[实验手册](https://cloud.tencent.com/developer/labs/lab/10518)

---

**1、实验环境：**

系统：win10 64bit  家庭版

GPU：RTX3070

CUDA版本：11.4

cuDNN版本：cuDNN v8.2.1 （for CUDA 11.x）

python：3.7.10

tensorflow-gpu：2.5.0

keras不用另外装，tf2内置了，代码要有部分修改

opencv-python：4.5.2.54

Pillow：8.2.0

**2、部署过程：**（仅记录重点部分、详细过程网上很多，说得非常详细，基本都是一路的next，装好后检查下环境变量配置，没自动加上的话要手动配上，我这安装好后基本都自动配置上了）

* 先安装visualstudio，因为装cuda时需要，2017或者2019都没问题，我装的2017社区版，免费就行。
  传送门：https://visualstudio.microsoft.com/zh-hans/downloads/
* 查看自己的显卡驱动版本信息中，cuda对应的版本，非常重要。
* tf2、py、cuDNN、cuda版本对应关系表（win版本）（非常重要，版本对应不上错误报到怀疑人生）：[cpu版](https://www.tensorflow.org/install/source_windows#cpu)、[gpu版](https://www.tensorflow.org/install/source_windows#gpu) （可能要梯子），例如我目前显卡驱动中显示cuda版本11.4.56，就选择11.x的cuda，8.x的cuDNN，2.4x+的tensorflow-gpu。
  原本我是这样理解的，但是实际中，tensorflow-gpu的版本最好选择对应支持的cuda和cuDNN版本中，最新的版本，例如我原本选择2.4.2的tensorflow-gpu就有问题，调用的cuda的lib不对，虽然可以根据错误提示下载对应的lib，但是比较麻烦。
* 安装cuda，cuda下载链接：https://developer.nvidia.com/cuda-downloads
* 安装cuDNN，cuDNN下载链接（需要注册）：https://developer.nvidia.com/rdp/cudnn-download

*nvidia的网站有时候非常抽筋，如果遇到怎么样也无法注册登录，或者登陆后怎么也打不开某个页面，换个时间再试，就算有个梯子也不行。*

* 安装 Anaconda，具体新增环境，切换环境，进入对应环境终端，操作过程不复杂，不再叙述。新建个python3.7的环境，然后切换到对应环境open terminal，接着后续操作。
* 实操时发现Anaconda中，tensorflow-gpu版本为2.3.0，我需要2.4.x+，建议终端中直接用pip安装，出问题的可能大大减少。
* 【非常重要】终端中使用pip时，注意是否连着梯子，连着的话建议关闭，否则pip会有奇怪报错。例如：`ValueError: check_hostname requires server_hostname`
* 安装tensorflow-gpu（新环境需要等很久，耐心等候）：pip install tensorflow-gpu==2.5.0

检查下安装是否成功，能否顺利调用到gpu，安装完tf2后，终端直接进入python，输入

```python
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpu)
```

正常无错误会打印如：`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

否则显示空：`[]`

* 安装opencv-python（如果后续有问题可以限制下版本，我当前默认装的4.5.2.54）：pip install opencv-python
* 安装Pillow：pip install Pillow==8.2.0
  如果不安装，本课程实验运行会报错：`ImportError: Could not import PIL.Image. The use of load_img  requires PIL.`
  另外如果不是按上述环境安装，之前装了旧版本的话，也可能报错。Pillow 7.x以上版本应该都是正常的。
  我实操时，默认最新版本8.3.0，会报错：`TypeError: __array__() takes 1 positional argument but 2 were given`
  查github的issues发现降级最容易解决，估计是与NumPy（1.19.5）有个兼容问题
* 安装scipy（我当前默认装的1.7.0）：pip install scipy
  如果不安装，实验会报错：`ImportError: Image transformations require SciPy. Install SciPy.`

**3、修改代码及运行**

　　至此，环境终于搭建好，正常到这应该没有什么坑的了。

　　原实验使用的tensorflow==1.14，安装了keras==2.3.1，tf2开始内置了keras，对应代码要有一点修改：

主要是引入文件部分：

```python
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
```
训练：python train.py

调整超参训练：python train.py --epochs 1

测试：python train.py --test 1 --resume_model_path ./results/model.h5