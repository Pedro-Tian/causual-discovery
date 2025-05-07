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

import castle
from castle.common import GraphDAG

from castle.datasets import DAG, IIDSimulation
from castle.algorithms import CORL, Notears, GOLEM, GraNDAG, DAG_GNN, GES
from dodiscover.toporder import SCORE, DAS, NoGAM, CAM

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
    log_path = f"./experiment_logs/v3_lamb_thresh_exp_{args.type}{args.h}N{args.nodes}_DCFAS_{args.model}.log"
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
    print(MBs)
    return MBs

def true_MB(true_dag, i):
    n_nodes = true_dag.shape[0]
    # print(f"====={i}====")
    parents = set(np.where(true_dag[:,i])[0])
    children = set(np.where(true_dag[i,:])[0])
    # print(f"parents:{parents}")
    # print(f"children:{children}")
    spouse = [i]
    for c in children:
        spouse += list(np.where(true_dag[:,c])[0])
    spouse = set(spouse)
    MB = sorted(parents.union(set(children), set(spouse)))
    # print(f"spouse:{spouse}")
    # print(f"MB:{MB}")
    return parents, children, spouse, MB

def eval_MB(tgt, pred):
    n = max(max(pred), max(tgt))+1
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
            sub_X_list.append(None)
            sub_true_dag_list.append(None)
            sub_nodes_list.append(None)
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

def merge_graph_voting(sub_nodes_list, sub_causal_matrix_list, true_dag):
    """
    A naiive voting merge method, directly weighted sum all edges mentioned in `sub_causal_matrix_list`

    sub_nodes_list: 2-D List
    sub_causal_matrix_list: List of np arrays
    """
    recover_graph = np.zeros(true_dag.shape)
    count = np.zeros(true_dag.shape)

    for nodes, sub_causal_matrix in zip(sub_nodes_list, sub_causal_matrix_list):
        # logger.info(f"[DEBUG] nodes {nodes}")
        # logger.info(f"[DEBUG] sub_causal_matrix {sub_causal_matrix}")
        if nodes is None: continue
        recover_graph[np.ix_(nodes, nodes)] += sub_causal_matrix
        count[np.ix_(nodes, nodes)] += 1
    
    count = np.maximum(count, np.ones(true_dag.shape))
    recover_graph = recover_graph/count
    
    return recover_graph

def merge_graph_voting_lamb(sub_nodes_list, sub_causal_matrix_list, true_dag, lamb=0.5):
    """
    A naiive voting merge method, directly weighted sum all edges mentioned in `sub_causal_matrix_list`

    sub_nodes_list: 2-D List
    sub_causal_matrix_list: List of np arrays
    """
    recover_graph = np.zeros(true_dag.shape)
    count = np.zeros(true_dag.shape)

    for nodes, sub_causal_matrix in zip(sub_nodes_list, sub_causal_matrix_list):
        if nodes is None: continue
        recover_graph[np.ix_(nodes, nodes)] += sub_causal_matrix
        count[np.ix_(nodes, nodes)] += 1
    
    coef = 1-np.exp(-lamb*count)
    count = np.maximum(count, np.ones(true_dag.shape))
    recover_graph = recover_graph/count*coef
    
    return recover_graph

def check_dag(arr):
    """
    arr np.array
    """
    G = nx.from_numpy_array(arr, create_using=nx.DiGraph())
    is_dag = nx.is_directed_acyclic_graph(G)
    return is_dag


def infer_causal(args, X, true_dag):
    causal_matrix_order, causal_matrix, met_before = None, None, None
    if args.model == 'CORL':
        # rl learn
        model = CORL(encoder_name='transformer',
                decoder_name='lstm',
                reward_mode='episodic',
                reward_regression_type='GPR',
                batch_size=64,
                input_dim=64,
                embed_dim=64,
                iteration=1000,
                device_type='gpu',
                device_ids=2)
        model.learn(X)
        causal_matrix = model.causal_matrix
    elif args.model == 'NOTEARS':
        model = Notears()
        model.learn(X)
        causal_matrix = model.causal_matrix
        causal_matrix_order = model.causal_matrix
    elif args.model == 'GES':
        model = GES()
        model.learn(X)
        causal_matrix = model.causal_matrix
        causal_matrix_order = model.causal_matrix
    elif args.model == 'GOLEM':
        model = GOLEM(num_iter=1e4, device_type='cpu')
        model.learn(X)
        causal_matrix = model.causal_matrix
        causal_matrix_order = model.causal_matrix
    elif args.model == 'DAGGNN':
        model = DAG_GNN()
        model.learn(X)
        causal_matrix = model.causal_matrix
        causal_matrix_order = model.causal_matrix
    elif args.model == 'GRANDAG':
        model = GraNDAG(input_dim=X.shape[1], iterations = 100000)
        model.learn(X)
        causal_matrix = model.causal_matrix
        causal_matrix_order = model.causal_matrix
    elif args.model == 'SCORE':
        # eta_G = 0.001
        # eta_H = 0.001
        # cam_cutoff = 0.001

        # causal_matrix, top_order_SCORE = SCORE(X, eta_G, eta_H, cam_cutoff)

        context = make_context().variables(data = pd.DataFrame(X)).build()
        model = SCORE()  # or DAS() or NoGAM() or CAM()
        # print("before learning")
        model.learn_graph(pd.DataFrame(X), context)
        # print("Finish learning")
        causal_matrix_order = nx.adjacency_matrix(model.order_graph_).todense()
        # print(causal_matrix_order, true_dag)
        try:
            met_before = castle.metrics.MetricsDAG(causal_matrix_order, true_dag)
        except Exception as e:
            met_before=None
            logger.info(f"[Error] met_before=None: {traceback.format_exc()}")
        causal_matrix = nx.adjacency_matrix(model.graph_).todense()
    elif args.model == 'DAS':
        context = make_context().variables(data = pd.DataFrame(X)).build()
        model = DAS() 
        model.learn_graph(pd.DataFrame(X), context)
        causal_matrix_order = nx.adjacency_matrix(model.order_graph_).todense()
        met_before = castle.metrics.MetricsDAG(causal_matrix_order, true_dag)
        causal_matrix = nx.adjacency_matrix(model.graph_).todense()
    elif args.model == 'CAM':
        context = make_context().variables(data = pd.DataFrame(X)).build()
        model = CAM() 
        model.learn_graph(pd.DataFrame(X), context)
        causal_matrix_order = nx.adjacency_matrix(model.order_graph_).todense()
        met_before = castle.metrics.MetricsDAG(causal_matrix_order, true_dag)
        causal_matrix = nx.adjacency_matrix(model.graph_).todense()
    elif args.model == 'NoGAM':
        context = make_context().variables(data = pd.DataFrame(X)).build()
        model = NoGAM() 
        model.learn_graph(pd.DataFrame(X), context)
        causal_matrix_order = nx.adjacency_matrix(model.order_graph_).todense()
        met_before = castle.metrics.MetricsDAG(causal_matrix_order, true_dag)
        causal_matrix = nx.adjacency_matrix(model.graph_).todense()

    return causal_matrix_order, causal_matrix, met_before

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
    lamb_choice = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 100]
    thresh_choice = [0, 0.05 ,0.1 ,0.15 ,0.2 ,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    lamb_and_thresh_exp_bef = {f"{l}_{t}": [] for l in lamb_choice for t in thresh_choice}
    lamb_and_thresh_exp_aft = {f"{l}_{t}": [] for l in lamb_choice for t in thresh_choice}
    logger.info(lamb_and_thresh_exp_aft)


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
        print(f"X: {X.shape}\n{X}")
        logger.info(f'true_dag\n{true_dag}')
        parents, children, spouse, sub_MB = true_MB(true_dag, 0)


        ## 测试MB结果
        t1 = time.time()
        markov_blankets = get_MB(X)
        time_MB = time.time()-t1
        

        # 根据 markov_blankets 分割 true_dag 和 X
        sub_X_list, sub_true_dag_list, sub_nodes_list = split_graph(markov_blankets, true_dag, X[:2000])

        time_subgraph_list = []
        sub_causal_matrix_list_befroe = []
        sub_causal_matrix_list_after = []
        mb_metrics_list = []

        

        for i, (sub_X, sub_true_dag, sub_nodes) in enumerate(zip(sub_X_list, 
                                                         sub_true_dag_list, 
                                                         sub_nodes_list)):
            logger.info(f"\n===  {i}-th graph ===")
            parents, children, spouse, sub_true_MB = true_MB(true_dag, i)
            logger.info(f"{len(sub_true_MB)} Nodes of True MB: {sub_true_MB}, parents: {parents}, children: {children}, spouse: {spouse}")
            if sub_nodes is None: 
                sub_causal_matrix_list_befroe.append(None)
                sub_causal_matrix_list_after.append(None)
                continue
            logger.info(f"{len(sub_nodes)} Nodes of Markov blanket: {sub_nodes}")
            mb_metrics_list.append(eval_MB(sub_true_MB, sub_nodes))
            
            t1 = time.time()
            sub_causal_matrix_order, sub_causal_matrix, sub_met2 = infer_causal(args, sub_X, sub_true_dag) 
            time_subgraph = time.time()-t1
            time_subgraph_list.append(time_subgraph)
            try:
                sub_met = castle.metrics.MetricsDAG(sub_causal_matrix, sub_true_dag)
            except Exception as e:
                sub_met = None
                logger.info(f"[Error] sub_met=None: {traceback.format_exc()}")
            # logger.info(f"\nsub_causal_matrix_order {type(sub_causal_matrix_order)}\n{sub_causal_matrix_order}")
            # logger.info(f"\nsub_causal_matrix {type(sub_causal_matrix)}\n{sub_causal_matrix}")
            logger.info(f"sub_met2 before prunning {sub_met2.metrics if sub_met2 else None}") 
            logger.info(f"sub_met after prunning {sub_met.metrics if sub_met else None}")
            
            # 剪枝前
            sub_causal_matrix_list_befroe.append(sub_causal_matrix_order)
            # 剪枝后
            sub_causal_matrix_list_after.append(sub_causal_matrix)

            # logger.info(f"sub_matrix_before\n{sub_causal_matrix_order.astype(np.int64)}")
            # logger.info(f"sub_matrix_after\n{sub_causal_matrix.astype(np.int64)}")
            # logger.info(f"sub_true_dag {type(sub_true_dag)}\n{sub_true_dag}")

        time_subgraph_avg = sum(time_subgraph_list)/len(time_subgraph_list) if len(time_subgraph_list)>0 else -1
        time_subgraph_max = max(time_subgraph_list)
        time_subgraph_tot = sum(time_subgraph_list)
        mb_eval = evaluation_summary(mb_metrics_list)
        logger.info(f"\n======= Graph Summary =======")
        logger.info(f"MB summary: {mb_eval}")
        # merge 
        try:
            t1 = time.time()
            merged_causal_matrix_bef = merge_graph_voting(sub_nodes_list, sub_causal_matrix_list_befroe, true_dag)
            merge_DAG = (GreedyFAS(merged_causal_matrix_bef)>0).astype(np.int64)
            time_FAS_before = time.time()-t1
            # np.save(f"./npy/{args.type}{args.h}N{args.nodes}_{args.model}_repeat{_}_before.npy", merge_DAG)
            merged_met_before = castle.metrics.MetricsDAG(merge_DAG, true_dag)
            logger.info(f"merged_met_before  before prunning {merged_met_before.metrics}")

            for l in lamb_choice:
                for t in thresh_choice:
                    merged_causal_matrix = merge_graph_voting_lamb(sub_nodes_list, sub_causal_matrix_list_befroe, true_dag, lamb=l)
                    merge_DAG = (GreedyFAS(merged_causal_matrix)>t).astype(np.int64)
                    merged_met = castle.metrics.MetricsDAG(merge_DAG, true_dag)
                    logger.info(f"merged_met_before lamb={l} thresh={t} {merged_met.metrics}")
                    lamb_and_thresh_exp_bef[f"{l}_{t}"].append(merged_met.metrics)
            
            # exit()
            t1 = time.time()
            merged_causal_matrix_aft = merge_graph_voting(sub_nodes_list, sub_causal_matrix_list_after, true_dag)
            merge_DAG = GreedyFAS(merged_causal_matrix_aft)
            merge_DAG = (merge_DAG>0).astype(np.int64)
            logger.info(merge_DAG)
            time_FAS_after = time.time()-t1
            # np.save(f"./npy/{args.type}{args.h}N{args.nodes}_{args.model}_repeat{_}_after.npy", merge_DAG)
            merged_met_after = castle.metrics.MetricsDAG(merge_DAG, true_dag)
            logger.info(f"merged_met_after  after prunning {merged_met_after.metrics}")

            for l in lamb_choice:
                for t in thresh_choice:
                    merged_causal_matrix = merge_graph_voting_lamb(sub_nodes_list, sub_causal_matrix_list_after, true_dag, lamb=l)
                    merge_DAG = (GreedyFAS(merged_causal_matrix)>t).astype(np.int64)
                    merged_met = castle.metrics.MetricsDAG(merge_DAG, true_dag)
                    logger.info(f"merged_met_after lamb={l} thresh={t} {merged_met.metrics}")
                    lamb_and_thresh_exp_aft[f"{l}_{t}"].append(merged_met.metrics)



            if merged_met_before:
                mmbef = merged_met_before.metrics
                mmbef['time_mb'] = time_MB
                mmbef['time_sub_avg'] = time_subgraph_avg
                mmbef['time_sub_max'] = time_subgraph_max
                mmbef['time_sub_tot'] = time_subgraph_tot
                mmbef['time_FAS'] = time_FAS_before
                mmbef['time_dist_tot'] = time_MB+time_subgraph_max+time_FAS_before
                mmbef['time_tot'] = time_MB+time_subgraph_tot+time_FAS_before
                res_before_prunning.append(mmbef)
            if merged_met_after:
                mmaft = merged_met_after.metrics
                mmaft['time_mb'] = time_MB
                mmaft['time_sub_avg'] = time_subgraph_avg
                mmaft['time_sub_max'] = time_subgraph_max
                mmaft['time_sub_tot'] = time_subgraph_tot
                mmaft['time_FAS'] = time_FAS_after
                mmaft['time_dist_tot'] = time_MB+time_subgraph_max+time_FAS_after
                mmaft['time_tot'] = time_MB+time_subgraph_tot+time_FAS_after
                res_after_prunning.append(mmaft)

        except Exception as e:
            logger.info(f"[Error] {traceback.format_exc()}")


    logger.info(lamb_and_thresh_exp_bef)
    for l in lamb_choice:
        for t in thresh_choice:
            summ = evaluation_summary(lamb_and_thresh_exp_bef[f"{l}_{t}"])
            logger.info(f"|lamb_thresh exp| before prunning lamb={l} thresh={t} met={summ}")
            summ = evaluation_summary(lamb_and_thresh_exp_aft[f"{l}_{t}"])
            logger.info(f"|lamb_thresh exp| after  prunning lamb={l} thresh={t} met={summ}")



    keys = res_after_prunning[0].keys()

    averages = {key: [] for key in keys}
    std_devs = {key: [] for key in keys}


    for dictionary in res_before_prunning:
        for key in keys:
            averages[key].append(dictionary[key])
            std_devs[key].append(dictionary[key])

    for key in keys:
        averages[key] = np.mean(averages[key])
        std_devs[key] = np.std(std_devs[key])

    result2 = {key: f"{averages[key]:.2f}+-{std_devs[key]:.2f}" for key in keys}
    logger.info(f"Before pruning: {result2}")


    averages = {key: [] for key in keys}
    std_devs = {key: [] for key in keys}


    for dictionary in res_after_prunning:
        for key in keys:
            averages[key].append(dictionary[key])
            std_devs[key].append(dictionary[key])

    for key in keys:
        averages[key] = np.mean(averages[key])
        std_devs[key] = np.std(std_devs[key])

    result = {key: f"{averages[key]:.2f}+-{std_devs[key]:.2f}" for key in keys}
    logger.info(f"After pruning: {result}")

