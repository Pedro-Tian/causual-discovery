# 运行目录
causual-discovery/main.py 是运行入口，不再使用causual-discovery/gcastle/main.py
# 运行方式
python main.py
# 环境路径变更
causual-discovery/gcastle/dodiscover/ 已被弃置，建议安装到环境内，避免相对路径调用
由于路径变更，可能需要安装gcastle和dodiscover
安装dodiscover时，在causual-discovery/dodiscover/ 目录下运行 pip install -e .
环境安装流程可以参考INSTALLATION.md，从零实现可运行环境
