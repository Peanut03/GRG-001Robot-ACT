# GRG 001机器人ACT

## 项目介绍

**V1.4版本**

**在收集数据record.py中添加了检查摄像头是否断开的报错警告**



**⚠️注意事项**

**检查是否为正确的摄像头格式（摄像头编号，格式，分辨率，帧率）**

**检查收集的数据是否正确**





**这是一个用于控制grg001机器人的ACT代码仓库，包含数据采集，数据清洗，训练，开环评估和闭环控制的代码**



**环境依赖**

可以直接使用lerobot环境，或者重新配置环境

```
conda env create -f conda_env.yaml
```



**录制数据**

先启动001机器人的主机摇操程序，确保采集设备和001机器人主机在**同一网络**下，**更改record_v3.py中的ip地址**，使用

```
v4l2-ctl --list-devices #查看摄像头编号
v4l2-ctl -d /dev/video0 --all #查看当前摄像头采集格式
v4l2-ctl -d /dev/video0 --list-formats-ext #查看摄像头可选采集格式
v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV #设置分辨率和格式
v4l2-ctl -d /dev/video0 --set-parm=30 #设置帧率

ffplay /dev/video（1-9）#查看正确的rgb摄像头
```

更改代码中的摄像头编号，然后进行数据采集，踩下踏板开始采集，松开踏板停止采集，录制一条数据

```
python record_v3.py
```



**清洗数据**

更改data_clean.py中的root_dir为正确的采集数据地址，运行

```
python data_clean.py
```

运行后每一条数据的文件夹下应该都有data_log_correct.txt文件



**验证数据及摇操复现**

启动001机器人数据驱动程序，修改为正确的ip地址，然后运行

```
cd test_scripts
python rundrive_test_001.py
```

可以看到机器人重现了这条数据的摇操



**数据转换**

grg001的手臂是8个自由度的（7个自由度手臂+1个自由度夹爪），值得注意的是手臂的每个关节都可以旋转到任意位置，因此前7个自由度需要使用mean_std进行归一化，而由于夹爪只有两个状态（0，255）而没有中间值，因此需要用min_max做归一化

训练act需要将数据转换成.h5格式，运行

``` 
pyhton convert_data.py  --input_dir your/cleaned_data/path --output_dir your/output_dir/path
```



**ACT训练**

训练配置文件在**act_pytorch/configs/basic.toml**下，可以修改自由度，如果是二分类自由度采用min_max归一化，训练结果保存在**experiments**文件夹下面

```
python train.py --dataset_dir converted_data
```



**ACT开环评估**

将一条数据的head和hand图像作为输入，查看模型输出的关节数据是否正常

```
python eval_local_images.py --checkpoint your/checkpoint/path --head_dir your/head_images/data --hand_dir your/hand_images/data
```

**ACT摄像头开环评估（用于排查错误）**

通过摄像头拍摄的图像进行开环评估，不会控制机器人

```
python camer_eval.py --checkpoint your/checkpoint/path
```

**ACT闭环控制**

打开001机器人的数据驱动程序，更改正确的ip地址，将两个摄像头的帧图像作为输入，将模型输出的关节数据传到001机器人进行闭环控制

```
python inference.py --checkpoint your/checkpoint/path
```
