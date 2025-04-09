1.该项目是一个基于神经网络的房价预测项目，可利用房价预测数据集进行房价预测模型店训练和测试

2.该项目运行需要一个conda环境和pytorch-cpu环境

3.安装pytroch cpu版
conda install pytorch torchvision torchaudio cpuonly -c pytorch
重要的事说三遍：
是cpu版！！！
是cpu版！！！
是cpu版！！！

4.安装其他依赖的库
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas

5.如果还有其他依赖的库没有安装会报错，根据报错安装上就可以了

6.网络的定义在net.py文件中，main.py文件中的主函数传参数可以指定网络的工作模式：train，训练模式；test，测试模式
