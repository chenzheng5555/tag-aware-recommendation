# Python

### 输入

```python
print("%s %5d"%("%输出",5))
print("{0}{1:5}".format("\"\".format()输出",5)) # 位置必须都给 或都不指定
print(f"f\"\"输出:{5:5}{'123'}")  # {}里不能用""?
```

# Keras

+ `keras.preprocessing.image.load_img(color_mode="rgb")`会自动将表示表示rgb转为rgb，源代码相关部分。

  ```python
  img = pil_image.open(path)
  ……
  elif color_mode == 'rgb':
      if img.mode != 'RGB':
          img = img.convert('RGB')
  ```

  

# TensorFlow





# Pytorch

## .utils.data.Dataset和.utils.data.DataLoader

```python
class DataBase(data.Dataset):
    def __init__(self, opt, train=True, datasplit=False)
        self.img_feature = h5py.File(os.path.join(opt.root, "insta_imgFeat.h5"), "r")
    def __getitem__(self, index)
        img_data = t.tensor(self.img_feature.get(img_id)).reshape((-1, 512))
        
dataset = DataBase(cfg)
dataloader = DataLoader(dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=4)
# 因为多线程会报错，报错原因可能各不相同，如
Exception has occurred:TypeError
h5py objects cannot be pickled
# 最简单的方法是将num_work设为0
# 修改1：将self.img_feature作为全局变量img_feature放在类外面，在本地笔记本上不报错，
# 但上传到服务器时读取img_feature.get出错
img_feature = h5py.File(os.path.join(opt.root, "insta_imgFeat.h5"), "r")
class DataBase(data.Dataset):
# 修改2：参考：https://zhuanlan.zhihu.com/p/101988536，在__getitem__里加载h5py.File，如
def __init__(self, opt, train=True, datasplit=False)
    self.img_feature = None
def __getitem__(self, index):
    if self.img_feature == None:
        self.img_feature = h5py.File(os.path.join(self.opt.root, "insta_imgFeat.h5"), "r")
# 
```

## Dataset，\_\_getitem\_\_()处理错误样本

```python
# 将出错的样本剔除。如果实在是遇到这种情况无法处理，则可以返回None对象，然后在Dataloader中实现自定义的collate_fn，将空对象过滤掉。但要注意，在这种情况下dataloader返回的batch数目会少于batch_size。
class NewDogCat(DogCat): # 继承前面实现的DogCat数据集
    def __getitem__(self, index):
        try:
            # 调用父类的获取函数，即 DogCat.__getitem__(self, index)
            return super(NewDogCat,self).__getitem__(index)
        except:
            return None, None

from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式
def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return t.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据
# 对于诸如样本损坏或数据集加载异常等情况，还可以通过其它方式解决。例如但凡遇到异常情况，就随机取一张图片代替：
class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            new_index = random.randint(0, len(self)-1)
            return self[new_index]
```

## 模型参数

```python
for name, param in model.named_parameters():
    print(name, param)  #param.size()
```

## 调整学习率

[参考](https://zhuanlan.zhihu.com/p/80876299)



# Torch_gometric



## MessagePassing

在`foward()`里调用`.propagate()`$\rightarrow$`propagate()`调用`.message()，.aggregate()，.update()`

```python
# 模型
def forward(self, x, edge_index):
    ……
    return self.propagate(edge_index, x=x, norm=norm)
# message_passing.py
def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
    ……
    out = self.message(**msg_kwargs)
    ……
    out = self.aggregate(out, **aggr_kwargs)
    ……
    return self.update(out, **update_kwargs)
```



# GPU

+ 参数模型训练和数据加载的时间差

```python
for ii, (data, label) in tqdm(enumerate(dataloader, start=1)):
    start = time.time()
    train()
    print(f"time cost:{time.time()-start}")
```

```cmd
# 没有GPU
cpu
0it [00:00, ?it/s]time cost:14.199122190475464
1it [00:20, 20.13s/it]time cost:11.57933783531189
2it [00:37, 19.32s/it]time cost:11.143985033035278
3it [00:54, 18.60s/it]time cost:11.598880290985107
4it [01:11, 18.19s/it]time cost:12.487136840820312
# 多线程加速后。
cpu
call DateBase:True
total posts:624520
test post:65057; train post:544469; samples:7497
99it [19:41, 11.42s/it]time cost:10.13833475112915,loss:26.557020396321604
133it [25:58, 10.67s/it]

# 有GPU
cuda:0
0it [00:00, ?it/s]time cost:2.0023834705352783
1it [00:09,  9.54s/it]time cost:0.10555744171142578
2it [00:18,  9.25s/it]time cost:0.10711979866027832
3it [00:26,  8.98s/it]time cost:0.11736178398132324
4it [00:34,  8.75s/it]time cost:0.10901737213134766
# 多线程加速后。
cuda:0
call DateBase:True
total posts:624520
test post:65057; train post:544469; samples:7497
99it [03:10,  1.85s/it]time cost:0.19095778465270996,loss:26.847382726729847
199it [06:17,  1.59s/it]time cost:0.20117545127868652,loss:104.93663821928203

# gpustat -i 600  >> frp.log 2>&1 &
node002              Sat Nov 14 22:28:11 2020  450.51.05
[0] TITAN RTX        | 44'C,   0 % |  5214 / 24220 MB | 10101002(5211M)
```







# SLURM集群

module load 共享模块后，会使用共享模块，而不会使用本地已有模块。

```cmd
#test.job文件里
module load python37
# 报错
Exception has occurred: ModuleNotFoundError
No module named 'torch'
# 注释掉后则会使用base环境里python解析器和模块
```



# 神经网络训练

+ 学习率设大了，

```cmd
# 学习率0.001 Dataloader shuffle=False adam
Epoch:0    	 acc:0.4226    ,prec:0.1182    ,rec:0.1451    ,f1:0.1019    
Epoch:1    	 acc:0.4300    ,prec:0.1284    ,rec:0.1595    ,f1:0.1121    
Epoch:2    	 acc:0.3450    ,prec:0.0892    ,rec:0.1066    ,f1:0.0762    
Epoch:3    	 acc:0.4399    ,prec:0.1288    ,rec:0.1570    ,f1:0.1113    
Epoch:4    	 acc:0.4563    ,prec:0.1307    ,rec:0.1634    ,f1:0.1139    
Epoch:5    	 acc:0.4534    ,prec:0.1300    ,rec:0.1640    ,f1:0.1135    
Epoch:6    	 acc:0.3644    ,prec:0.1069    ,rec:0.1344    ,f1:0.0934    
Epoch:7    	 acc:0.3649    ,prec:0.1093    ,rec:0.1405    ,f1:0.0965    
Epoch:8    	 acc:0.4254    ,prec:0.1299    ,rec:0.1657    ,f1:0.1146    
Epoch:9    	 acc:0.4658    ,prec:0.1452    ,rec:0.1851    ,f1:0.1280    
Epoch:10   	 acc:0.3299    ,prec:0.0885    ,rec:0.1165    ,f1:0.0785    
Epoch:11   	 acc:0.3777    ,prec:0.1069    ,rec:0.1324    ,f1:0.0932    
Epoch:12   	 acc:0.2266    ,prec:0.0598    ,rec:0.0796    ,f1:0.0538    
Epoch:13   	 acc:0.2310    ,prec:0.0716    ,rec:0.0955    ,f1:0.0648    
Epoch:14   	 acc:0.2593    ,prec:0.0849    ,rec:0.1094    ,f1:0.0754    
Epoch:15   	 acc:0.2902    ,prec:0.0924    ,rec:0.1189    ,f1:0.0820    
Epoch:16   	 acc:0.2050    ,prec:0.0632    ,rec:0.0796    ,f1:0.0558    
Epoch:17   	 acc:0.2168    ,prec:0.0678    ,rec:0.0852    ,f1:0.0598    
Epoch:18   	 acc:0.2529    ,prec:0.0770    ,rec:0.0956    ,f1:0.0676
# 学习率修改为0.0001 shuffle=True torch.adam
Epoch:0    	 acc:0.3785    ,prec:0.1054    ,rec:0.1208    ,f1:0.0881    
Epoch:1    	 acc:0.4794    ,prec:0.1592    ,rec:0.1878    ,f1:0.1369      
Epoch:11   	 acc:0.6101    ,prec:0.2355    ,rec:0.2978    ,f1:0.2113    
Epoch:12   	 acc:0.6177    ,prec:0.2391    ,rec:0.3023    ,f1:0.2145    
Epoch:13   	 acc:0.6197    ,prec:0.2411    ,rec:0.3041    ,f1:0.2162    
Epoch:14   	 acc:0.6234    ,prec:0.2432    ,rec:0.3083    ,f1:0.2184    
Epoch:15   	 acc:0.6254    ,prec:0.2436    ,rec:0.3082    ,f1:0.2189    
Epoch:16   	 acc:0.6198    ,prec:0.2429    ,rec:0.3053    ,f1:0.2176    
Epoch:17   	 acc:0.6177    ,prec:0.2410    ,rec:0.3043    ,f1:0.2161
```

