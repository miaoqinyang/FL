import argparse, json
import datetime
import os
import logging
import torch, random

from torch.utils.data import DataLoader
from core.models import resnet18
from core.client.trainer import Client
from core.algrorithm.fedavg import fedavg
from core.dataset import cifar10

# print(dataset)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')  # 定义命令行解析器对象
    parser.add_argument('-c', '--conf', dest='conf')  # 添加命令行参数
    args = parser.parse_args()  # 从命令行中结构化解析参数

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)
    # 准备数据集
    lrp = resnet18.get_model("resnet18")

    train_datasets, eval_datasets = cifar10.get_dataset("./data/", conf["type"])

    algrorithm = fedavg(conf,lrp,eval_datasets)

    clients = []

    for c in range(conf["clients"]):
        clients.append(Client(conf, lrp, train_datasets, c))
        # print(Client(conf, server.global_model, train_datasets, c).local_train(server.global_model).keys())
    print("\n\n")
    for e in range(conf["global_epochs"]):
        candidates = random.sample(clients, conf["k"])

        weight_accumulator = {}

        for name, params in lrp.state_dict().items():

            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff = c.local_train(lrp)
            # print(diff)
            for name, params in lrp.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        algrorithm.model_aggregate(weight_accumulator)
        acc, loss = algrorithm.model_eval()
        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
