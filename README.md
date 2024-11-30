在kagggle上面有一个cafa5的比赛，可以利用[proteinglm-1b-mlm](https://huggingface.co/Bo1015/proteinglm-1b-mlm)来进行测试。

使用[proteinglm-1b-mlm](https://huggingface.co/Bo1015/proteinglm-1b-mlm)模型来对创建测试和训练数据的蛋白质嵌入。

之后利用上面的提供的开源eval文件进行评测，在A800gpu上。

项目地址：

https://github.com/SteinsHimmer/cafa5.git

步骤流程：

1.下载https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data里面的Data数据集并保存在根目录下。

2.创建相应的环境，并安装相应的库,Python版本 3.10.15（请看requirement.txt）：

```
pip install -r requirements.txt
```

3.运行`generate_embeddings.py`,之后获得相应的npy文件。

4.运行`eval.ipynb`，然后取得相应评估结果(替换里面的npy文件。）

5.本来还想在A800测试一下，但本人服务器最近占用率较高，eval.py属于未完成)



![](https://gitlab.com/sky6445638/picture/-/raw/master/pictures/2024/11/30_22_50_51_202411302250853.png)

![](https://gitlab.com/sky6445638/picture/-/raw/master/pictures/2024/11/30_22_51_18_202411302251022.png)

