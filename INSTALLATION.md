# 从零配置环境
conda create -n py310 python==3.10
pip install --upgrade pip

## 安装gcastle等
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gcastle numpy scipy torch pygam

## 安装cdt
pip install cdt

## 安装dodiscover
git clone https://github.com/py-why/dodiscover.git
### cd 到克隆目录下
pip install -e .

## DCILP依赖包
pip install igraph

# 运行孙神代码
python main.py