# GRG 001机器人ACT

## 项目介绍

**V2.0版本**

**重构了data_load，排查了延迟，频率问题，act模型验证成功**



**这是一个用于控制grg001机器人的ACT代码仓库，包含数据采集，数据清洗，训练，开环评估和闭环控制的代码**

**⚠️注意事项**

**检查是否为正确的摄像头格式（摄像头编号，格式，分辨率，帧率）**

**检查收集的数据是否正确**



**摇操采集数据**

```
进入第4个，倒数第3个  3.18内核

密码1234

cd /home/grg/ros/lylong7/new/grg7dof-alg/

source install/setup.bash

roslaunch grgrobot_control grgrobot.launch
```



**然后连接到jetson orin agx**

```
#连接grg-visitor网络 密码grg166125
ssh grg@10.1.157.161
#连接dr.frank网络 密码Gundam00
ssh grg@192.168.50.32


密码 123456
sudo su 密码123456

cd /Users/act_pytorch-main/
```

**环境依赖**

可以直接使用lerobot环境，或者重新配置环境

```
conda activate lerobot
```



**录制数据**

先启动001机器人的主机摇操程序，确保采集设备和001机器人主机在**同一网络**下，**更改record_v4.py中的ip地址**，ip为**192.168.50.83**，使用

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
cd tools/collect_data
python record_v4.py #加入了摄像头断开连接到报错
```





**清洗数据**

更改data_clean.py中的root_dir为正确的采集数据地址，运行

```
python tools/data_clean.py 
#将50hz采集到数据下采样到5hz
python tools/get_new_data.py
```

运行后每一条数据的文件夹下应该都有data_log_correct.txt文件

使用new_data进行训练



**数据转换**

grg001的手臂是8个自由度的（7个自由度手臂+1个自由度夹爪），值得注意的是手臂的每个关节都可以旋转到任意位置，因此前7个自由度需要使用mean_std进行归一化，夹爪的范围在[0,255]之间，用 *2/255 -1 映射到[-1,1]。

训练act需要将数据转换成.h5格式，运行

``` 
cd tools/
pyhton convert_data.py  --input_dir your/cleaned_data/path 
```



启动机器人主机，选择第二个进阶选择中的第三个内核

**机器人主机密码及操作**

```
g002152
cd /home/grg/xlm/3.0.1.3/body_control #三个终端 su 密码g002152
./operate.sh start com
./operate.sh start data_bridge
./operate.sh start
```



**验证数据及摇操复现**

启动001机器人数据驱动程序，修改为正确的ip地址，然后运行，ip为**192.168.50.83**

```
cd tools/test_scripts/
#在这个目录下放一个data_log_correct.txt的数据集
python rundrive_test_001.py
```

可以看到机器人重现了这条数据的摇操



**ACT训练**

连接到服务器

```
#连接intranet
ssh songwei@10.1.47.59
密码 123456
将数据保存到/home/space2/下
#激活环境
conda activate lerebot
#创建experiments文件夹
mkdir experiments
```



训练配置文件在**act_pytorch/configs/basic.toml**下，可以修改自由度，训练结果保存在experiments文件夹下面

```
python train.py --dataset_dir tools/converted_data
```



**测试websockets通信延迟**

```
python tools/websockets_receive_delay.py #测试接收通信延迟
python tools/websockets_send_delay.py #测试发送通信延迟
```



**查看接收到的手臂和夹爪数据**

```
python tools/get_arm_status.py #手臂
python tools/get_hand_status.py #夹爪
```



**ACT开环评估**

将一条数据的head和hand图像作为输入，查看模型输出的关节数据是否正常

```
python eval_local_images.py --checkpoint your/checkpoint/path --head_dir your/head_images/data --hand_dir your/hand_images/data
```

**ACT摄像头开环评估（用于排查错误）**

通过摄像头拍摄的图像进行开环评估，不会控制机器人

```
python camera_eval.py --checkpoint your/checkpoint/path
```

打开001机器人的数据驱动程序，更改正确的ip地址，将两个摄像头的帧图像作为输入，将模型输出的关节数据传到001机器人进行闭环控制

**ACT闭环控制**

```
python inference_test.py --checkpoint your/checkpoint/path
```





