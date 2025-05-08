# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This demo script aim to demonstrate
how to use CORL algorithm in `castle` package for causal inference.

If you want to plot causal graph, please make sure you have already install
`networkx` package, then like the following import method.

Warnings: This script is used only for demonstration and cannot be directly
          imported.
"""

import os
os.environ['CASTLE_BACKEND'] ='pytorch'
import pandas as pd
import networkx as nx
import numpy as np
import traceback
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import matplotlib.pyplot as plt
from cdt.causality.graph import GS, IAMB, Fast_IAMB, Inter_IAMB, MMPC

import castle
from castle.common import GraphDAG

from castle.datasets import DAG, IIDSimulation
# from castle.algorithms import CORL, Notears, GOLEM, GraNDAG, DAG_GNN
from dodiscover.toporder import SCORE#, DAS, NoGAM, CAM

from dodiscover.context_builder import make_context

from dcilp.dcdilp_Ph1MB1 import *
from dcilp.dcdilp_Ph1MB1 import _threshold_hard, _MBs_fromInvCov
import dcilp.utils_files.utils as utils
from dcilp.utils_files.gen_settings import gen_data_sem_original

from mas_approximation import MAS_Approx
from merge import adjacency_matrix_to_dag, GreedyFAS

import argparse
import logging


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'causal-discovery')))


def set_logger(args):
    # 配置日志
    # 记录不同级别的日志
    # logger.debug("这是一条debug信息")
    # logger.info("这是一条info信息")
    # logger.warning("这是一条warning信息")
    # logger.error("这是一条error信息")
    # logger.critical("这是一条critical信息")
    log_path = f"./experiment_logs/MBexp_v.10_N{args.nodes}{args.type}{args.h}_num{args.num_observation}.log"
    if os.path.exists(log_path): os.remove(log_path)
    
    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG
    logger.setLevel(log_level)

    log_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # log_formatter = logging.Formatter('%(message)s')
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(log_level)
    # console_handler.setFormatter(log_formatter)
    # logger.addHandler(console_handler)
    
    return logger
    
    

def get_MB(data, ice_lam_min = 0.01, ice_lam_max = 0.1, ice_lam_n = 40):
    # 这三个参数维持SCILP默认设置，具体取值也许论文里提及了？ TODO double check hyper-parameters in DCILP paper
    # ice_lam_min, ice_lam_max, ice_lam_n 0.1, 0.3, 10

    data = data - np.mean(data, axis=0, keepdims=True)
    # Method ICE empirical

    t0 = timer()
    out = ice_sparse_empirical(data, lams=np.linspace(ice_lam_min, \
                                                    ice_lam_max, \
                                                    ice_lam_n))
    Theta = _threshold_hard(out[0], tol=1e-3)

    MBs = _MBs_fromInvCov(Theta)
    t1 = timer()
    print(t1-t0)
    # print(MBs)
    return MBs


def compute_MB(X, method='GS'):
    assert method in ['GS', 'IAMB', 'Fast_IAMB', 'Inter_IAMB', 'MMPC']
    X = pd.DataFrame(X)
    # print(X.head(), X.shape)
    print(f'running {method}')
    t1 = time.time()
    if method == 'GS':
        obj = GS()
        print("before predict")
        output = obj.predict(X)
        adj_matrix = nx.adjacency_matrix(output).todense()
    if method == 'IAMB':
        obj = IAMB()
        output = obj.predict(X)
        adj_matrix = nx.adjacency_matrix(output).todense()
    if method == 'Fast_IAMB':
        obj = Fast_IAMB()
        output = obj.predict(X)
        adj_matrix = nx.adjacency_matrix(output).todense()
    if method == 'Inter_IAMB':
        obj = Inter_IAMB()
        output = obj.predict(X)
        adj_matrix = nx.adjacency_matrix(output).todense()
    if method == 'MMPC':
        obj = MMPC()
        output = obj.predict(X)
        adj_matrix = nx.adjacency_matrix(output).todense()
    # nx.draw_networkx(output, font_size=8)
    # plt.savefig(f'mb_graph_{method}.jpg')
    return adj_matrix, time.time()-t1



def true_MB(true_dag, i):
    n_nodes = true_dag.shape[0]
    # print(f"====={i}====")
    parents = set(np.where(true_dag[:,i])[0])
    children = set(np.where(true_dag[i,:])[0])
    # print(f"parents:{parents}")
    # print(f"children:{children}")
    spouse = []
    for c in children:
        spouse += list(np.where(true_dag[:,c])[0])
    spouse = set(spouse)
    MB = sorted(parents.union(set(children), set(spouse)))
    # print(f"spouse:{spouse}")
    # print(f"MB:{MB}")
    return parents, children, spouse, MB

def eval_MB(tgt, pred):
    # print(pred, tgt)
    max_p = 0 if len(pred) == 0 else max(pred)
    max_g = 0 if len(tgt) == 0 else max(tgt)
    n = max(max_p, max_g)+1
    p = [0]*n
    g = [0]*n
    for i in pred: p[i] = 1
    for i in tgt: g[i] = 1

    precision = precision_score(g,p)
    recall = recall_score(g,p)
    acc = accuracy_score(g,p)
    f1 = f1_score(g,p)
    logger.info(f"[MB metrics] precision: {precision:.3f}, recall: {recall:.3f}, acc: {acc:.3f}, f1: {f1:.3f}")
    return {'precision': precision, 'recall': recall, 'acc': acc, 'f1': f1}

def evaluation_summary(list_of_dic):
    keys = list_of_dic[0].keys()
    averages = {key: [] for key in keys}
    std_devs = {key: [] for key in keys}
    for dictionary in list_of_dic:
        for key in keys:
            averages[key].append(dictionary[key])
            std_devs[key].append(dictionary[key])

    for key in keys:
        averages[key] = np.mean(averages[key])
        std_devs[key] = np.std(std_devs[key])

    result = {key: f"{averages[key]:.2f}+-{std_devs[key]:.2f}" for key in keys}
    return result

def split_graph(markov_blankets, true_dag, X):
    # sub_X_list: 每个元素为子图对应的数据矩阵
    # sub_true_dag_list: 每个元素为子图对应的邻接矩阵 (感觉好像没用)
    # sub_nodes_list：很重要，子图排序完之后要恢复回原来的节点

    sub_X_list = []
    sub_true_dag_list = []
    sub_nodes_list = []
    n_nodes = len(markov_blankets)

    for i in range(n_nodes):
        blanket_indices = np.where(markov_blankets[i])[0]
        # print(i, blanket_indices)
        if len(blanket_indices) <= 1:
            sub_X_list.append([])
            sub_true_dag_list.append([])
            sub_nodes_list.append([])
            continue
        # 把节点 i 自己也加进去
        nodes = set(blanket_indices)
        nodes.add(i)
        nodes = sorted(nodes)

        sub_X = X[:, nodes]

        sub_dag = true_dag[np.ix_(nodes, nodes)]

        sub_X_list.append(sub_X)
        sub_true_dag_list.append(sub_dag)
        sub_nodes_list.append(nodes)

    return sub_X_list, sub_true_dag_list, sub_nodes_list     


def check_dag(arr):
    """
    arr np.array
    """
    G = nx.from_numpy_array(arr, create_using=nx.DiGraph())
    is_dag = nx.is_directed_acyclic_graph(G)
    return is_dag



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='causal inference')

    parser.add_argument('--model', default='SCORE', type=str, 
                        help='model name')
    parser.add_argument('--nodes', default=10, type=int,
                        help="number of nodes")
    parser.add_argument('--h', default=2, type=int,
                        help="number of edges")
    parser.add_argument('--type', default='ER', type=str,
                        help="type of graph")
    parser.add_argument('--method', default='linear', type=str,
                        help="?")
    parser.add_argument('--sem_type', default='gauss', type=str,
                        help="?")
    parser.add_argument('--repeat', default=10, type=int,
                        help="number of repeated iterations")
    parser.add_argument('--num_observation', default=2000, type=int,
                        help="number of observation data")

    args = parser.parse_args()

    # setup log
    logger = set_logger(args)
    # logger.setLevel(logging.DEBUG)

    # type = 'ER'  # or `SF`
    # h = 5  # ER2 when h=5 --> ER5
    # n_nodes = 50
    n_edges = args.h * args.nodes
    # method = 'linear'
    # sem_type = 'gauss'

    res_after_prunning = []
    res_before_prunning = []



    print(args.type, args.h, args.nodes, args.model)

    for _ in range(args.repeat):
        logger.info(f"\n================ repeat number {_+1} ====================")
        if args.type == 'ER':
            weighted_random_dag = DAG.erdos_renyi(n_nodes=args.nodes, n_edges=n_edges,
                                                weight_range=(0.5, 2.0), seed=1000+_*100)
        elif args.type == 'SF':
            weighted_random_dag = DAG.scale_free(n_nodes=args.nodes, n_edges=n_edges,
                                                weight_range=(0.5, 2.0), seed=1000+_*100)
        else:
            raise ValueError('Just supported `ER` or `SF`.')

        dataset = IIDSimulation(W=weighted_random_dag, n=args.num_observation,
                                method=args.method, sem_type=args.sem_type)
        true_dag, X = dataset.B, dataset.X
        # print(f"X: {X.shape}\n{X}")
        logger.info(f'true_dag\n{true_dag}')
        parents, children, spouse, sub_MB = true_MB(true_dag, 0)


        ## 测试MB结果
        t1 = time.time()
        markov_blankets = get_MB(X)
        # print(len(markov_blankets))
        time_MB = time.time()-t1
        
        # com_mb = compute_MB(X)
        # mb_methods = ['GS', 'IAMB', 'Fast_IAMB', 'Inter_IAMB', 'MMPC']
        mb_methods = []
        mbs = {k: compute_MB(X, method=k) for k in mb_methods}
        mb_metrics_methods = {k: [] for k in mb_methods}
        # 根据 markov_blankets 分割 true_dag 和 X
        sub_X_list, sub_true_dag_list, sub_nodes_list = split_graph(markov_blankets, true_dag, X)
        # print(len(sub_nodes_list))
        # print(sub_nodes_list)
        # exit()

        mb_metrics_list = []
        
        for i, (sub_X, sub_true_dag, sub_nodes) in enumerate(zip(sub_X_list, 
                                                         sub_true_dag_list, 
                                                         sub_nodes_list)):
            logger.info(f"\n===  {i}-th graph ===")
            parents, children, spouse, sub_true_MB = true_MB(true_dag, i)
            logger.info(f"{len(sub_true_MB)} Nodes of True MB: {sub_true_MB}, parents: {parents}, children: {children}, spouse: {spouse}")
            mb_metrics_list.append({**eval_MB(sub_true_MB, sub_nodes), **{'time': time_MB}})
            logger.info(f"{len(sub_nodes)} Nodes of DCILP MB: {sub_nodes}")

            for k in mb_methods:
                com_mb = mbs[k][0]
                com_parents, com_children, com_spouse, sub_com_MB = true_MB(com_mb, i)
                logger.info(f"{len(sub_com_MB)} Nodes of {k} MB: {sub_com_MB}, parents: {com_parents}, children: {com_children}, spouse: {com_spouse}")
                mb_metrics_methods[k].append({**eval_MB(sub_true_MB, sub_com_MB), **{'time': mbs[k][1]}})


        mb_eval = evaluation_summary(mb_metrics_list)
        logger.info(f"\n======= Graph Summary =======")
        logger.info(f"DCILP MB summary: {mb_eval}")
        for k in mb_methods:
            mb_eval = evaluation_summary(mb_metrics_methods[k])
            logger.info(f"{k} MB summary: {mb_eval}")
        # merge 




