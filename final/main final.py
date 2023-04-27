from scapy.all import *
from scapy.arch.windows import get_windows_if_list
from scapy.layers.inet import IP, TCP, UDP

import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


batch_size = 4000
lr = 0.002
num_epoch = 500
number_of_features = 41
number_of_class_attack_cat = 10
number_of_class_label = 2
hyper_parameter1=128
hyper_parameter2=64
hyper_parameter3=32
hyper_parameter4=16

# GPU OR CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

class DataSet_UNSWNB15_attack_cat(Dataset):

    def __init__(self, data_set_path):
        df = pd.read_csv(data_set_path)

        # preprocceing
        df.drop(['id', "service"], axis=1, inplace=True)
        cols = ['proto', 'state']
        df[cols] = df[cols].apply(LabelEncoder().fit_transform)


        self.data_array = df.values
        print(self.data_array)
        self.x = self.data_array[:, :41]
        # self.y = self.data_array[:, 41]

        # normalaized data
        scaler = preprocessing.StandardScaler().fit(self.x)
        self.x = scaler.transform(self.x)

        self.x = torch.tensor(self.x).float()
        # self.y = torch.tensor(self.y).long()

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, item):
        sample = (self.x[item, :], self.y[item])
        return sample


class DataSet_UNSWNB15_label(Dataset):

    def __init__(self, data_set_path):
        df = pd.read_csv(data_set_path)

        # preprocceing
        df.drop(['id', "service"], axis=1, inplace=True)
        cols = ['proto', 'state']
        df[cols] = df[cols].apply(LabelEncoder().fit_transform)

        self.data_array = df.values
        self.x = self.data_array[:, :41]
        # self.y = self.data_array[:, 41]

        # normalaized data
        scaler = preprocessing.StandardScaler().fit(self.x)
        self.x = scaler.transform(self.x)

        self.x = torch.tensor(self.x).float()
        # self.y = torch.tensor(self.y).long()

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, item):
        sample = (self.x[item, :], self.y[item])
        return sample


class Model_Attack_Cat(nn.Module):

    def __init__(self):
        super(Model_Attack_Cat, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(number_of_features, hyper_parameter1),
            nn.Tanh(),
            nn.Linear(hyper_parameter1, hyper_parameter2),
            nn.Tanh(),
            nn.Linear(hyper_parameter2, hyper_parameter3),
            nn.Sigmoid(),
            nn.Linear(hyper_parameter3, hyper_parameter4),
            nn.Sigmoid(),
            nn.Linear(hyper_parameter4, number_of_class_attack_cat)
        )

    def forward(self, x):
        return self.layers(x)


class Model_Label(nn.Module):

    def __init__(self):
        super(Model_Label, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(number_of_features, hyper_parameter1),
            nn.Tanh(),
            nn.Linear(hyper_parameter1, hyper_parameter2),
            nn.Tanh(),
            nn.Linear(hyper_parameter2, hyper_parameter3),
            nn.Sigmoid(),
            nn.Linear(hyper_parameter3, hyper_parameter4),
            nn.Sigmoid(),
            nn.Linear(hyper_parameter4, number_of_class_label)
        )

    def forward(self, x):
        return self.layers(x)


def extract_features(packet):
    features = {}
    if IP not in packet:
        return None
    features['id'] = packet[IP].id
    features['dur'] = packet.time - packet[IP].time
    features['proto'] = packet[IP].proto

    if TCP in packet:
        features['service'] = 'tcp'
        features['state'] = packet[TCP].flags
    elif UDP in packet:
        features['service'] = 'udp'
        features['state'] = '-'
    else:
        return None

    features['spkts'] = packet[IP].len - packet[IP].ihl * 4 - len(packet[TCP] if TCP in packet else packet[UDP])
    features['dpkts'] = len(packet[TCP] if TCP in packet else packet[UDP])
    features['sbytes'] = packet[IP].len - packet[IP].ihl * 4 - len(packet[TCP] if TCP in packet else packet[UDP])
    features['dbytes'] = len(packet[TCP] if TCP in packet else packet[UDP])

    features['rate'] = features['spkts'] / features['dur']
    features['sttl'] = packet[IP].ttl
    features['dttl'] = packet[TCP].window if TCP in packet else packet[UDP].chksum

    features['sload'] = features['sbytes'] / features['dur']
    features['dload'] = features['dbytes'] / features['dur']

    features['sloss'] = 0 if TCP in packet else packet[IP].frag
    features['dloss'] = packet[IP].tos

    features['sinpkt'] = 0 if features['spkts'] == 0 else features['dur'] / features['spkts']
    features['dinpkt'] = 0 if features['dpkts'] == 0 else features['dur'] / features['dpkts']

    features['sjit'] = 0 if features['spkts'] == 0 else sum(
        abs(features['sinpkt'] - features['dur'] / features['spkts']) for i in range(1, features['spkts'])) / (
                                                                    features['spkts'] - 1)
    features['djit'] = 0 if features['dpkts'] == 0 else sum(
        abs(features['dinpkt'] - features['dur'] / features['dpkts']) for i in range(1, features['dpkts'])) / (
                                                                    features['dpkts'] - 1)

    features['swin'] = packet[TCP].window if TCP in packet else 0
    features['stcpb'] = packet[TCP].options[0][1] if TCP in packet and len(packet[TCP].options) > 0 else 0
    features['dtcpb'] = packet[TCP].options[1][1] if TCP in packet and len(packet[TCP].options) > 1 else 0
    features['dwin'] = packet[TCP].options[2][1] if TCP in packet and len(packet[TCP].options) > 1 else 0

    if TCP in packet:
        features['tcprtt'] = packet[TCP].options[3][1] if len(packet[TCP].options) > 3 else 0
        features['synack'] = int('S' in features['state'] and 'A' in features['state'])
        features['ackdat'] = int('A' in features['state'] and 'S' not in features['state'])
        features['smean'] = packet[TCP].sport
        features['dmean'] = packet[TCP].dport
        features['trans_depth'] = 0
        features['response_body_len'] = 0
        features['ct_srv_src'] = 0
        features['ct_state_ttl'] = 0
        features['ct_dst_ltm'] = 0
        features['ct_src_dport_ltm'] = 0
        features['ct_dst_sport_ltm'] = 0
        features['ct_dst_src_ltm'] = 0
        features['is_ftp_login'] = 0
        features['ct_ftp_cmd'] = 0
        features['ct_flw_http_mthd'] = 0
        features['ct_src_ltm'] = 0
        features['ct_srv_dst'] = 0 # ---
        features['is_sm_ips_ports'] = 0 #--

    elif UDP in packet:
        features['tcprtt'] = 0
        features['synack'] = 0
        features['ackdat'] = 0
        features['smean'] = packet[UDP].sport
        features['dmean'] = packet[UDP].dport
        features['trans_depth'] = 0
        features['response_body_len'] = 0
        features['ct_srv_src'] = 0
        features['ct_state_ttl'] = 0
        features['ct_dst_ltm'] = 0
        features['ct_src_dport_ltm'] = 0
        features['ct_dst_sport_ltm'] = 0
        features['ct_dst_src_ltm'] = 0
        features['is_ftp_login'] = 0
        features['ct_ftp_cmd'] = 0
        features['ct_flw_http_mthd'] = 0
        features['ct_src_ltm'] = 0
        features['ct_srv_dst'] = 0 # ---
        features['is_sm_ips_ports'] = 0   # ---

    # for key in features:
    #     print(features[key],end=",")

    df = pd.DataFrame([features])
    df.to_csv('my_file.csv', index=False, header=True)
    print(df)
    # -----------------------------------------------------------
    dataset_attack_cat = DataSet_UNSWNB15_attack_cat("my_file.csv")
    # dataset_label = DataSet_UNSWNB15_label("my_file.csv")

    PATH = f'model_attack.pt'
    my_model_attack = Model_Attack_Cat().to(device=device)
    model_state_dict = torch.load(PATH).to(device=device)
    my_model_attack.load_state_dict(model_state_dict)

    print(my_model_attack)

    test_attack_cat_dataloader = DataLoader(dataset=dataset_attack_cat, batch_size=1, shuffle=True)


def udp_sniffer():
    """start a sniffer.
    """
    # interfaces = get_windows_if_list()
    # print(interfaces)

    # print('\n[*] start udp sniffer')
    sniff(
        filter="",
        iface=r'Intel(R) Dual Band Wireless-AC 8260', prn=extract_features
    )


if __name__ == '__main__':
    udp_sniffer()


