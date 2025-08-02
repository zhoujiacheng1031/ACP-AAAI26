# -*- coding: utf-8 -*-
"""
Created on 2024-07-31

@file: get_concept.py
@purpose: get concept from neo4j    
"""

from py2neo import Graph,Node,Relationship
from data_utils import *
import pdb
import json
import torch
import argparse

graph = Graph('neo4j://115.156.114.150:27687',user='neo4j',password='dasineo4j')

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default=None, help='seed training set')
parser.add_argument('--test_path', default=None, help='test set')
parser.add_argument('--dev_path', default=None, help='seed dev set')
parser.add_argument('--con2id_path', default='', help='concept2id file')
parser.add_argument('--save_num', type=int, default=600, help='number of concepts with highest frequent')

args = parser.parse_args()

def get_entity_concept(e_str, topk=1):
    
    try:
        res = graph.run("MATCH (i:Instance {name:\"" + e_str + "\"})-[r:IS_A]->(c:Concept) RETURN i.name AS Instance, tofloat(r.probability)/10000 AS `is a(n)`, c.name AS Concept ORDER BY `is a(n)` DESC LIMIT " + str(topk) + ";").data()
    except:
        return ""
    
    if len(res) == 0:
        return ""
    else:
        return res[0]['Concept']
    
def get_data_concept(in_file, out_file, con_dict=None):
    data_list = get_data_list(in_file)
    if con_dict is None:
        con_dict = {}
    with open(out_file, 'w', encoding='utf-8') as outf:

        total_count = 0
        no_con_count = 0
        for data in data_list:
            h_str = data['h']['name'].split(' ')[-1].lower()
            h_con = get_entity_concept(h_str)

            t_str = data['t']['name'].split(' ')[-1].lower()
            t_con = get_entity_concept(t_str)

            data['h']['concept'] = h_con
            data['t']['concept'] = t_con

            if h_con not in con_dict:
                con_dict[h_con] = 0
            con_dict[h_con] += 1

            if t_con not in con_dict:
                con_dict[t_con] = 0
            con_dict[t_con] += 1

            outf.write(json.dumps(data))
            outf.write('\n')

            if h_con == '':
                no_con_count += 1
            if t_con == '':
                no_con_count += 1
            total_count += 2
            print('no concept {} / total concept {}'.format(no_con_count, total_count))

    return con_dict

def dump_con2id(con2id_file, con_dict, save_num):
    sort_con_dict = sorted(con_dict.items(), key=lambda x: x[1], reverse=True)
    with open(con2id_file, 'w', encoding='utf-8') as conf:
        con2id = {}
        idx = 0
        for item in sort_con_dict:
            con = item[0]
            if len(con) == 0:
                continue
            con2id[con] = idx
            idx += 1
            if idx >= save_num:
                break
        conf.write(json.dumps(con2id))

if __name__ == '__main__':
    if args.train_path is not None:
        con_dict = get_data_concept(args.train_path, args.out_train_path)
    if args.dev_path is not None:
        con_dict = get_data_concept(args.dev_path, args.out_dev_path, con_dict)
    if args.test_path is not None:
        con_dict = get_data_concept(args.test_path, args.out_test_path, con_dict)

    dump_con2id(args.con2id_path, con_dict, args.save_num)