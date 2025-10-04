'''

file structure

-data
    -__MACOSX
        -train
            -0
            -16
            -2
            -21
            -22
            -8
    -data126282
        -text.zip
        -train.zip
    -train
        -0
            -img_1.jpg
            -...
        -1
        -...
        -40
    -eval.txt
    -garbage_dict.json
    -readme.json
    -train.txt
-output
-visualdl_log_dir
-work
-3437279.ipynb
-infer_image.jpg
-main.ipynb

'''

# 加载相关库类
import os
import zipfile
import random
import json
import paddle
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddle.io import Dataset
import paddle.vision.transforms as T


'''
参数配置
'''
train_parameters = {
    "input_size": [3, 244, 244],                              #输入图片的shape
    "class_dim": -1,                                          #分类数
    "src_path":"/home/aistudio/data/data126282/train.zip",    #原始数据集路径
    "target_path":"/home/aistudio/data/",             #要解压的路径
    "train_list_path": "/home/aistudio/data/train.txt",       #train.txt路径
    "eval_list_path": "/home/aistudio/data/eval.txt",         #eval.txt路径
    "readme_path": "/home/aistudio/data/readme.json",         #readme.json路径
    "label_dict":{},                                          #标签字典
    "num_epochs": 1,                                          #训练轮数
    "train_batch_size": 4,                                    #训练时每个批次的大小
    "skip_steps": 10,
    "save_steps": 300, 
    "learning_strategy": {                                    #优化函数相关的配置
        "lr": 0.0001                                          #超参数学习率
    },
    "checkpoints": "/home/aistudio/work/checkpoints"          #保存的路径

}


# 数据准备
def unzip_data(src_path,target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    '''
    if(not os.path.isdir(target_path + "train")):     
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def get_data_list(target_path,train_list_path,eval_list_path):
    '''
    生成数据列表
    '''
    #存放所有类别的信息
    class_detail = []
    #获取所有类别保存的文件夹名称
    data_list_path=target_path+"train/"
    class_dirs = os.listdir(data_list_path)  
    #总的图像数量
    all_class_images = 0
    #存放类别标签
    class_label=0
    #存放类别数目
    class_dim = 0
    #存储要写进eval.txt和train.txt中的内容
    trainer_list=[]
    eval_list=[]
    #读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            #每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            #统计每个类别有多少张图片
            class_sum = 0
            #获取类别路径 
            path = data_list_path  + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:                                  # 遍历文件夹下的每个图片
                if img_path.split(".")[-1] == "jpg":
                    name_path = path + '/' + img_path                       # 每张图片的路径
                    if class_sum % 8 == 0:                                  # 每8张图片取一个做验证数据
                        eval_sum += 1                                       # test_sum为测试数据的数目
                        eval_list.append(name_path + "\t%d" % class_label + "\n")
                    else:
                        trainer_sum += 1 
                        trainer_list.append(name_path + "\t%d" % class_label + "\n")#trainer_sum测试数据的数目
                    class_sum += 1                                          #每类图片的数目
                    all_class_images += 1                                   #所有类图片的数目
                else:
                    continue
            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir             #类别名称
            class_detail_list['class_label'] = class_label          #类别标签
            class_detail_list['class_eval_images'] = eval_sum       #该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
            class_detail.append(class_detail_list)  
            #初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1 
            
    #初始化分类数
    train_parameters['class_dim'] = class_dim
  
    #乱序  
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image) 
            
    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image) 

    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path                  #文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'],'w') as f:
        f.write(jsons)
    print ('生成数据列表完成！')



'''
参数初始化
'''
src_path=train_parameters['src_path']
target_path=train_parameters['target_path']
train_list_path=train_parameters['train_list_path']
eval_list_path=train_parameters['eval_list_path']

'''
解压原始数据到指定路径
'''
unzip_data(src_path,target_path)

'''
划分训练集与验证集，乱序，生成数据列表
'''
#每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
with open(eval_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
    
#生成数据列表   
get_data_list(target_path,train_list_path,eval_list_path)

print(os.listdir(target_path))



class Reader(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.img_paths = []
        self.labels = []

        if mode == 'train':
            with open(os.path.join(self.data_path, "train.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label)) # 训练模式下获取训练数据路径及label

        else:
            with open(os.path.join(self.data_path, "eval.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label)) # 测试模式下获取测试数据路径及label

    # 在__getitem__中定义对图片数据的读取、数据的预处理等操作。
    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 文件索引号
        :return:
        """
        # 第一步打开图像文件并获取label值
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB') 
        img = img.resize((224, 224), Image.BILINEAR)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1)) / 255
        label = self.labels[index]
        label = np.array([label], dtype="int64")
        return img, label 

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)




# 模型配置
# 为构建VGG网络，先定义卷积池化操作类，该类继承自paddle.nn.Layer
class ConvPool(paddle.nn.Layer):
    '''卷积+池化'''
    # 构造函数，进行网络以及一些参数的设定及初始化
    def __init__(self,
                 num_channels, # 1输入数据的通道数（如RGB图像为3通道）
                 num_filters,  # 2卷积核的数量，决定输出特征的维度
                 filter_size, # 3卷积核的尺寸（如3表示3×3的卷积核）
                 pool_size, # 4池化窗口的大小
                 pool_stride, # 5池化操作的步长
                 groups, # 6连续卷积层的重复次数
                 conv_stride=1, # 卷积操作的步长（默认1）
                 conv_padding=1, # 填充大小（默认1，保持特征图尺寸不变）
                 ):
        super(ConvPool, self).__init__()  # 网络层构建

        # 循环创建groups个卷积层和ReLU激活层
        for i in range(groups):
            self.add_sublayer(   #添加子层实例
                'bb_%d' % i,
                paddle.nn.Conv2D(         # layer，paddle.nn.Conv2D是二维卷积层，用于提取空间特征
                in_channels=num_channels, # 输入通道数
                out_channels=num_filters,   # 输出通道数（卷积核个数）
                kernel_size=filter_size,   # 卷积核大小
                stride=conv_stride,        # 步长
                padding = conv_padding,    # padding边缘填充
                )
            )
            self.add_sublayer(
                'relu%d' % i,
                paddle.nn.ReLU() # 修正线性单元激活函数，引入非线性特性
            )
            num_channels = num_filters # 每次循环后更新num_channels，确保下一层的输入通道数与当前层输出通道数一致
            
        # 添加最大池化层
        self.add_sublayer(
            'Maxpool',
            paddle.nn.MaxPool2D(             #二维最大池化层，用于降维和特征压缩
            kernel_size=pool_size,           #池化核大小
            stride=pool_stride               #池化步长
            )
        )

    # 执行函数，定义网络的前向执行逻辑，执行groups个连续卷积+一个池化操作
    # 显式向前传播
    def forward(self, inputs):
        x = inputs # 接收输入数据
        for prefix, sub_layer in self.named_children(): # 遍历所有子层，named_children()返回所有子层的名称和实例
            # print(prefix,sub_layer)
            x = sub_layer(x) # 将当前数据传递给子层进行处理
        return x # 返回最终处理结果
        


# 使用前面定义的ConvPool来构造VGG16网络（13卷积层+3全连接层）
class VGGNet(paddle.nn.Layer):
  
    def __init__(self):
        super(VGGNet, self).__init__()   
        # 5组卷积-池化层    
        self.convpool01 = ConvPool(
            3, 64, 3, 2, 2, 2)  #3:通道数，64：卷积核个数，3:卷积核大小，2:池化核大小，2:池化步长，2:连续卷积个数
        self.convpool02 = ConvPool(
            64, 128, 3, 2, 2, 2)
        self.convpool03 = ConvPool(
            128, 256, 3, 2, 2, 3) 
        self.convpool04 = ConvPool(
            256, 512, 3, 2, 2, 3)
        self.convpool05 = ConvPool(
            512, 512, 3, 2, 2, 3)       
        self.pool_5_shape = 512 * 7* 7
        # 3层全连接层
        self.fc01 = paddle.nn.Linear(self.pool_5_shape, 4096)
        self.fc02 = paddle.nn.Linear(4096, 4096)
        self.fc03 = paddle.nn.Linear(4096, train_parameters['class_dim']) # 分类层，train_parameters是分类的类别数量

    def forward(self, inputs, label=None):
        # print('input_shape:', inputs.shape) #[8, 3, 224, 224]
        """前向计算"""
        out = self.convpool01(inputs)
        # print('convpool01_shape:', out.shape)           #[8, 64, 112, 112]
        out = self.convpool02(out)
        # print('convpool02_shape:', out.shape)           #[8, 128, 56, 56]
        out = self.convpool03(out)
        # print('convpool03_shape:', out.shape)           #[8, 256, 28, 28]
        out = self.convpool04(out)
        # print('convpool04_shape:', out.shape)           #[8, 512, 14, 14]
        out = self.convpool05(out)
        # print('convpool05_shape:', out.shape)           #[8, 512, 7, 7]         
        # 逐步提取从低级到高级的图像特征
        batch_size = out.shape[0]
        out = paddle.reshape(out, shape=[batch_size, -1]) # reshape之后才能作为全连接层的输入！！特征图展平为一维向量，-1自动计算批次大小
        out = self.fc01(out) # 25088->4096
        out = self.fc02(out) # 4096->4096
        out = self.fc03(out) # 4096->类别数
        
        if label is not None:
            acc = paddle.metric.accuracy(input=out, label=label) # 计算预测结果
            return out, acc
        else:
            return out
        # 训练模式提供label，返回预测结果和准确率，预测模式无label只返回预测结果



# 模型训练

print(train_parameters['class_dim'])
print(train_parameters['label_dict'])


#高层API
# 定义输入
input_define = paddle.static.InputSpec(shape=[-1, 3 , 244, 244],
                                   dtype="float32",
                                   name="img")

label_define = paddle.static.InputSpec(shape=[-1, 1],
                                       dtype="int64",
                                       name="label")  

model = VGGNet()
model = paddle.Model(model, inputs=input_define, labels=label_define) # 高层API的模型包装器，提供训练、评估等高级功能
params_info = model.summary((1,3,244,244)) # 生成模型的详细结构信息
print(params_info) # 打印模型基础结构和参数信息

# 实例化网络
model = VGGNet()
model = paddle.Model(model, inputs=input_define, labels=label_define)


optimizer = paddle.optimizer.Adam(learning_rate=train_parameters['learning_strategy']['lr'],
                                  parameters=model.parameters()) # 优化器，常用的梯度下降算法
# 模型准备
model.prepare(optimizer=optimizer,
                loss=paddle.nn.CrossEntropyLoss(),   # 损失函数使用交叉熵，
                metrics=paddle.metric.Accuracy())    # 评价指标使用准确率

callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir') # 可视化回调

# 这里的Reader是刚刚已经定义好的，代表训练数据
model.fit(train_data=Reader(data_path='/home/aistudio/data', mode='train'),
            eval_data=Reader(data_path='/home/aistudio/data', mode='eval'),
            batch_size=16,
            epochs=18,
            save_dir="output/",
            save_freq=5,       #保存模型的频率，多少个 epoch 保存一次模型
            log_freq=20,     #日志打印的频率，多少个 step 打印一次日志
            shuffle=True,callbacks=callback)



# 模型评估
# 实例化模型对象并加载模型参数
model_eval = paddle.Model(VGGNet(), inputs=input_define, labels=label_define) # 创建了一个空的模型框架，具有VGG16的网络结构，但参数尚未初始化或加载。
model_eval.load('output/final') # 从指定路径加载预训练的模型参数

model_eval.prepare(metrics=paddle.metric.Accuracy())    # 评价指标使用准确率
# 进行模型评估 数据读取器实例-估数据集存储路径-指定使用评估模式的数据集
result = model_eval.evaluate(eval_data=Reader(data_path='/home/aistudio/data', mode='eval'),log_freq=8,batch_size=32)



# 模型预测
class InferDataset(Dataset): # 与定义Reader的方式相同
    def __init__(self, img_path=None):
        """
        数据读取Reader(推理)
        :param img_path: 推理单张图片
        """
        super().__init__()
        if img_path:
            self.img_paths = [img_path]
        else:
            raise Exception("请指定需要预测对应图片路径")

    def __getitem__(self, index):
        # 获取图像路径
        img_path = self.img_paths[index]
        # 使用Pillow来读取图像数据并转成Numpy格式
        img = Image.open(img_path)
        if img.mode != 'RGB': 
            img = img.convert('RGB') 
        img = img.resize((224, 224), Image.ANTIALIAS) # 抗锯齿，高质量但较慢
        img = np.array(img).astype('float32') 
        img = img.transpose((2, 0, 1)) / 255  # HWC to CHW 并像素归一化

        return img

    def __len__(self):
        return len(self.img_paths)

        
label_dic = train_parameters['label_dict']
print(label_dic)
infer_path='infer_image.jpg'
infer_img = Image.open(infer_path)
#根据数组绘制图像
plt.imshow(infer_img)          
#显示图像
plt.show()                    
# 实例化推理模型
model = paddle.Model(VGGNet(), inputs=input_define)
# 读取刚刚训练好的参数
model.load("./output/final")
# 初始化模型
model.prepare()
# 实例化InferDataset
infer_data = InferDataset(infer_path)
# 进行推理
result = model.predict(test_data=infer_data)[0]
print(result)
result = paddle.to_tensor(result)
result = paddle.nn.functional.softmax(result)
print(result)
lab = np.argmax(result.numpy())
print("预测结果为：{:}, 概率为:{:.2f}" .format(label_dic[str(lab)],result.numpy()[0][0][lab]))



