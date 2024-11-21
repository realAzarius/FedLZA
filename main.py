import os
import sys
import argparse
import numpy as np
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import ConcatModel, CombinedModel
from Enterprise import Enterprise, EnterprisesInNetwork
# FedAnil+: Consortium Blockchain
from Block import Block
# FedAnil+: Consortium Blockchain
from Consortium_Blockchain import Consortium_Blockchain
import warnings

warnings.filterwarnings('ignore')

# set program execution time for logging purpose
date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_files_folder_path = f"logs/{date_time}"
NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"
from Enterprise import flcnt, lastprc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Block_FedAvg_Simulation")

'''
debug attributes 调试属性
gpu 使用的gpu id 0，1，2，3 默认为0
verbose 调试日志的开关 默认为1 开启
save_network_snapshots 是否保存网络快照 默认为0，不保存
destroy_tx_in_block 是否销毁存储在区块中的交易节省gpu内存，默认为0，不销毁
resume_path 从指定的网络快照路径恢复 默认为None
save_freq 保存网络快快照的频率 默认为5
save_most_recent 保存最近指定数量的数量，0表示保存所有 默认为2
'''
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0,
                    help='only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder')
parser.add_argument('-dtx', '--destroy_tx_in_block', type=int, default=0,
                    help='currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.')
parser.add_argument('-rp', '--resume_path', type=str, default=None,
                    help='resume from the path of saved network_snapshots; only provide the date')
parser.add_argument('-sf', '--save_freq', type=int, default=5, help='save frequency of the network_snapshot')
parser.add_argument('-sm', '--save_most_recent', type=int, default=2,
                    help='in case of saving space, keep only the recent specified number of snapshops; 0 means keep all')

'''
FL attributes 联邦学习属性
batchsize 本地训练批次大小 默认为10
model_name：要训练的模型名称。默认为 'OARF'
learning_rate：学习率，使用原始论文中的值作为默认值。默认为 0.01
optimizer：使用的优化器，默认为 SGD（随机梯度下降)
IID：数据分配给企业的方式。默认为 0，即非独立同分布
max_num_comm：最大通信轮数，如果提前收敛可能会提前终止。默认为 100
num_enterprises：模拟网络中的企业数量。默认为 20
shard_test_data：是否分片测试数据。默认为 0，即不分片
num_malicious：网络中恶意企业的数量。恶意企业的数据集将引入高斯噪声。默认为 0
noise_variance：注入的高斯噪声的方差水平。默认为 1
default_local_epochs：本地训练的轮数。默认为 5
'''
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='OARF', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01,
                    help="learning rate, use value from origin paper as default")
parser.add_argument('-op', '--optimizer', type=str, default="SGD",
                    help='optimizer to be used, by default implementing stochastic gradient descent')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to enterprises')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=100,
                    help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-nd', '--num_enterprises', type=int, default=20,
                    help='numer of the enterprises in the simulation network')
parser.add_argument('-st', '--shard_test_data', type=int, default=0,
                    help='it is easy to see the global models are consistent across enterprises when the test dataset is NOT sharded')
parser.add_argument('-nm', '--num_malicious', type=int, default=0,
                    help="number of malicious enterprises in the network. Malicious Enterprises data sets will be introduced Gaussian noise")
parser.add_argument('-nv', '--noise_variance', type=int, default=1,
                    help="noise variance level of the injected Gaussian Noise")
parser.add_argument('-le', '--default_local_epochs', type=int, default=5,
                    help='local train epoch. Train local model by this same num of epochs for each local_enterprise, if -mt is not specified')

'''
FedAnil+: Consortium_blockchain system consensus attributes 联盟区块链系统共识属性
unit_reward：提供数据、验证签名等的单位奖励。默认为 1
knock_out_rounds：如果一个企业被识别为恶意企业，将在该轮数后被踢出企业的对等列表。默认为 6
lazy_local_enterprise_knock_out_rounds：如果一个本地企业在该轮数内未提供更新（由于太慢或懒惰），将被踢出企业的对等列表。默认为 10
pow_difficulty：挖矿难度，如果设置为 0，则表示矿工使用 PoS（权益证明）。默认为 0
'''
parser.add_argument('-ur', '--unit_reward', type=int, default=1,
                    help='unit reward for providing data, verification of signature, validation and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6,
                    help="a local_enterprise or validator enterprise is kicked out of the enterprise's peer list(put in black list) if it's identified as malicious for this number of rounds")
parser.add_argument('-lo', '--lazy_local_enterprise_knock_out_rounds', type=int, default=10,
                    help="a local_enterprise enterprise is kicked out of the enterprise's peer list(put in black list) if it does not provide updates for this number of rounds, due to too slow or just lazy to do updates and only accept the model udpates.(do not care lazy validator or miner as they will just not receive rewards)")
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="if set to 0, meaning miners are using PoS")

'''
FedAnil+: Consortium_blockchain FL validator/miner restriction tuning parameters 联盟区块链验证器，矿工限制调整参数
miner_acception_wait_time：矿工接受交易的默认时间窗口，单位为秒。0 表示无时间限制。默认为 0.0
miner_accepted_transactions_size_limit：矿工接受的交易大小限制。0 表示无大小限制。默认为 0.0
miner_pos_propagated_block_wait_time：从通信轮次开始计算的等待时间，用于模拟 PoS 中的分叉事件。默认为无穷大
validator_threshold：用于确定恶意本地企业的准确率差异阈值。默认为 1.0
malicious_updates_discount：对被投票为负面的本地企业的更新应用折扣因子，而不是完全丢弃。默认为 0.0
malicious_validator_on：允许恶意验证者翻转投票结果。默认为 0
'''
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0,
                    help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each enterprise will just perform same amount(-le) of epochs per round like in FedAvg paper")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0,
                    help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size")
parser.add_argument('-mp', '--miner_pos_propagated_block_wait_time', type=float, default=float("inf"),
                    help="this wait time is counted from the beginning of the comm round, used to simulate forking events in PoS")
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0,
                    help="a threshold value of accuracy difference to determine malicious local_enterprise")
parser.add_argument('-md', '--malicious_updates_discount', type=float, default=0.0,
                    help="do not entirely drop the voted negative local_enterprise transaction because that risks the same local_enterprise dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative local_enterprise's updates are by some rate applied so it won't repeat")
parser.add_argument('-mv', '--malicious_validator_on', type=int, default=0,
                    help="let malicious validator flip voting result")

'''
distributed system attributes 分布式系统属性
network_stability：企业在线的概率。默认为 1.0
even_link_speed_strength：用于模拟传输延迟的链接速度强度。默认为 1，即每个企业被分配相同的链接速度强度。如果设置为 0，则链接速度强度在 0 和 1 之间随机初始化
base_data_transmission_speed：当 -els == 1 时，每秒可传输的数据量。默认为 70000.0
even_computation_power：用于模拟硬件设备强度的计算能力。默认为 1，即均匀分配计算能力。如果设置为 0，则计算能力随机初始化为 0 到 4 之间的整数
'''
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds a enterprise is online')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1,
                    help="This variable is used to simulate transmission delay. Default value 1 means every enterprise is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a enterprise will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0,
                    help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed (bandwidth), which further determines the transmission delay - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1,
                    help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")

'''
simulation attributes 模拟属性
hard_assign：在网络中硬分配角色的数量，按照本地企业、验证器和矿工的顺序。例如 "12,5,3" 分配 12 个本地企业、5 个验证器和 3 个矿工。",,*" 表示在每个通信轮次中完全随机分配角色
all_in_one：在注册时让所有节点都了解彼此。默认为 1
check_signature：如果设置为 0，则所有签名都假定为已验证，以节省执行时间。默认为 1
attack_type：设置用于攻击模拟的攻击类型。默认为空
target_acc：设置模拟结束时的目标准确率。默认为 0.9
'''
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*',
                    help="hard assign number of roles in the network, order by local_enterprise, validator and miner. e.g. 12,5,3 assign 12 local_enterprises, 5 validators and 3 miners. \"*,*,*\" means completely random role-assigning in each communication round ")
parser.add_argument('-aio', '--all_in_one', type=int, default=1,
                    help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-cs', '--check_signature', type=int, default=1,
                    help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('-at', '--attack_type', type=str, default='', help='set the attack type used for attack simulation')
parser.add_argument('-ta', '--target_acc', type=float, default=0.9, help='set the target accuracy for end simulation')
# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')

'''
python3 main.py -nd 100 -max_ncomm 50 -ha 80,10,10 -aio 1 -pow 0 -ko 5 -nm 3 -vh 0.08 -cs 0 -B 64 -mn OARF -iid 0 -lr 0.01 -dtx 1 -le 20
nd 100 100个客户端
max_ncomm 50 最大50轮通信
ha 80,10,10 角色分配每轮通信80个worker，10个验证器，10个矿工
aio 1 模拟中的每一个企业在peer list总都有其他企业
pow 0 工作量证明的难度，当使用0时，FedAnil+与FedAnil+-PoS共识一起运行，以选择获胜的矿工。
ko 5 一个企业在连续6轮被识别为恶意工人后，就会被列入黑名单。
nm 3 恰好有3家企业会成为恶意节点。
vh 0.08 在所有通信轮中，Validator-threshold设置为0.08。在未来的版本中，验证器可能会自适应地学习这个值。
cs 0 由于模拟没有包括干扰事务数字签名的机制，该参数关闭了签名检查以加快执行。
B 64 批量大小
mn OARF 使用OARF数据集
iid 0 以非iid的方式对训练数据集进行分片。
lr 0.01 学习率0.01
dtx 1 销毁存储在区块中的交易节省gpu内存
le 20 默认本地轮数20
'''

if __name__ == "__main__":
    # create logs/ if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # get arguments
    args = parser.parse_args()
    args = args.__dict__

    # detect CUDA  选择gpu或者cpu
    # dev = torch.device("cpu")
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # pre-define system variables 通信轮次？
    latest_round_num = 0

    ''' If network_snapshot is specified, continue from left 网络快照恢复模拟环境'''
    # 网络快照的恢复路径
    if args['resume_path']:
        # 如果没有设置保存网络快照，输出提示信息
        if not args['save_network_snapshots']:
            print("NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing.")
        network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args['resume_path']}"
        # 获取最新的网络快照文件名
        latest_network_snapshot_file_name = \
            sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')],
                   key=lambda fn: int(fn.split('_')[-1]), reverse=True)[0]
        print(f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}")
        print("BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel")
        latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
        enterprises_in_network = pickle.load(
            open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
        """获取企业列表"""
        enterprises_list = list(enterprises_in_network.enterprises_set.values())
        log_files_folder_path = f"logs/{args['resume_path']}"
        # for colab
        # log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
        # original arguments file
        args_used_file = f"{log_files_folder_path}/args_used.txt"
        file = open(args_used_file, "r")
        log_whole_text = file.read()
        lines_list = log_whole_text.split("\n")
        for line in lines_list:
            # abide by the original specified rewards
            if line.startswith('--unit_reward'):
                rewards = int(line.split(" ")[-1])
            # get number of roles
            if line.startswith('--hard_assign'):
                roles_requirement = line.split(" ")[-1].split(',')
            # get mining consensus
            if line.startswith('--pow_difficulty'):
                mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'
        # determine roles to assign
        try:
            local_enterprises_needed = int(roles_requirement[0])
        except:
            local_enterprises_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1
    else:
        ''' SETTING UP FROM SCRATCH'''
        '''没有指定恢复路径时从头开始模拟环境的初始状态'''

        # 0. create log_files_folder_path if not resume
        os.mkdir(log_files_folder_path)

        # 1. save arguments used 将当前运行时所用的命令行参数保存到日志文件args_used.txt中
        with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
            f.write("Command line arguments used -\n")
            f.write(' '.join(sys.argv[1:]))  # 传递给python脚本的参数
            f.write("\n\nAll arguments used -\n")
            for arg_name, arg in args.items():
                f.write(f'\n--{arg_name} {arg}')

        # 2. create network_snapshot folder 创建网络快照文件夹
        if args['save_network_snapshots']:
            network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
            os.mkdir(network_snapshot_save_path)

        # 3. assign system variables
        # for demonstration purposes, this reward is for every rewarded action
        # 每次奖励操作的基本奖励单位
        rewards = args["unit_reward"]  # 1

        # 4. get number of roles needed in the network
        # 解析 hard_assign 参数，提取网络所需的角色数目。 80 10 10 worker,验证器，矿工
        roles_requirement = args['hard_assign'].split(',')

        # determine roles to assign
        try:
            local_enterprises_needed = int(roles_requirement[0])  # 80个本地企业数量
        except:
            local_enterprises_needed = 1
        try:
            validators_needed = int(roles_requirement[1])  # 10个验证器数量
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])  # 10个矿工数量
        except:
            miners_needed = 1

        # 5. check arguments eligibility 检查参数的合理性

        num_enterprises = args['num_enterprises']  # 企业总数 100
        num_malicious = args['num_malicious']  # 恶意总数 3

        if num_enterprises < local_enterprises_needed + miners_needed + validators_needed:
            # 若企业总数不足，报错
            sys.exit(
                "ERROR: Roles assigned to the enterprises exceed the maximum number of allowed enterprises in the network.")

        if num_enterprises < 3:
            # 网络中至少需要三个企业 一个本地企业，一个矿工，一个验证者
            sys.exit(
                "ERROR: There are not enough enterprises in the network.\n The system needs at least one miner, one local_enterprise and/or one validator to start the operation.\nSystem aborted.")

        if num_malicious:
            if num_malicious > num_enterprises:
                # 若恶意企业数量大于企业数量
                sys.exit(
                    "ERROR: The number of malicious enterprises cannot exceed the total number of enterprises set in this network")
            else:
                print(
                    f"Malicious enterprises vs total enterprises set to {num_malicious}/{num_enterprises} = {(num_malicious / num_enterprises) * 100:.2f}%")

        # 6. create neural net based on the input model name 创建神经网络模型
        # net = None
        # net = ConcatModel()
        net = CombinedModel()
        # if args['model_name'] == 'cnn':
        #	net = CNN()
        # elif args['model_name'] == 'OARF':
        #	net = ConcatModel()

        # 7. assign GPU(s) if available to the net, otherwise CPU 分配设备CPU或GPU
        # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        # 到时候在服务器训练的时候可以改成cuda，在本地进行测试的时候先mps，后面的依次要改
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # if torch.cuda.device_count() > 1 :#or device:
        #	net = torch.nn.DataParallel(net)
        # print(f"{torch.cuda.device_count()} GPUs are available to use!")
        # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
        print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        print(f"Is MPS available? {torch.backends.mps.is_available()}")

        net = net.to(dev)

        # 统计 MPS 设备数量，但是代码有问题，这里是字符串的长度3
        num_mps_devices = len(device)
        print("Number of MPS devices: ", num_mps_devices)
        print("\n")

        # 8. set loss_function 设置损失函数
        loss_func = F.cross_entropy

        # 9. create enterprises in the network 创建网络中的企业
        enterprises_in_network = EnterprisesInNetwork(data_set_name='femnist',
                                                      is_iid=args['IID'],
                                                      batch_size=args['batchsize'],
                                                      learning_rate=args['learning_rate'],
                                                      loss_func=loss_func,
                                                      opti=args['optimizer'],
                                                      num_enterprises=num_enterprises,
                                                      network_stability=args['network_stability'],
                                                      net=net,
                                                      dev=dev,
                                                      knock_out_rounds=args['knock_out_rounds'],
                                                      lazy_local_enterprise_knock_out_rounds=args[
                                                          'lazy_local_enterprise_knock_out_rounds'],
                                                      shard_test_data=args['shard_test_data'],
                                                      miner_acception_wait_time=args['miner_acception_wait_time'],
                                                      miner_accepted_transactions_size_limit=args[
                                                          'miner_accepted_transactions_size_limit'],
                                                      validator_threshold=args['validator_threshold'],
                                                      pow_difficulty=args['pow_difficulty'],
                                                      even_link_speed_strength=args['even_link_speed_strength'],
                                                      base_data_transmission_speed=args['base_data_transmission_speed'],
                                                      even_computation_power=args['even_computation_power'],
                                                      malicious_updates_discount=args['malicious_updates_discount'],
                                                      num_malicious=num_malicious,
                                                      noise_variance=args['noise_variance'],
                                                      check_signature=args['check_signature'],
                                                      not_resync_chain=args['destroy_tx_in_block'])

        del net  # 删除net变量，net变量已经传递给EnterpriseInNetwork对象，现在已经不需要，释放内存
        enterprises_list = list(enterprises_in_network.enterprises_set.values())  # 所有企业节点的集合

        # 10. register enterprises and initialize global parameters 模拟网络中注册企业，初始化全局模型,peer_list
        for enterprise in enterprises_list:
            # set initial global weights 设定一个初始的全局模型参数
            enterprise.init_global_parameters()
            # helper function for registration simulation - set enterprises_list and aio    all_in_one :1
            enterprise.set_enterprises_dict_and_aio(enterprises_in_network.enterprises_set, args["all_in_one"])
            # simulate peer registration, with respect to enterprise idx order 模拟企业注册到网络
            enterprise.register_in_the_network()
        # remove its own from peer list if there is 从对等节点peer list中移除自身
        for enterprise in enterprises_list:
            enterprise.remove_peers(enterprise)

        """查看企业的peer list"""
        # for enterprise in enterprises_list:
        #     print()
        #     print('This is {}'.format(enterprise.return_idx()))
        #     for peer in enterprise.peer_list:
        #         print(peer.return_idx(), end=' ')

        # 11. build logging files/database path 在指定的日志文件路径下创建多个日志文件，记录网络中各种类型的事件
        # create log files
        # 正确剔除网络中的本地节点，恶意节点
        open(f"{log_files_folder_path}/correctly_kicked_local_enterprises.txt", 'w').close()
        # 被剔除本地企业节点，跟踪哪些企业节点被错误的识别并剔除
        open(f"{log_files_folder_path}/mistakenly_kicked_local_enterprises.txt", 'w').close()
        # 误报的恶意节点，视为恶意节点但是未被剔除
        open(f"{log_files_folder_path}/false_positive_malicious_nodes_inside_slipped.txt", 'w').close()
        # 没有被正确识别的节点，误认为恶意的良好节点
        open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'w').close()
        # open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'w').close()
        # open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'w').close()
        # 因为懒惰(未积极参与网络活动或训练的节点)被剔除的本地节点
        open(f"{log_files_folder_path}/kicked_lazy_local_enterprises.txt", 'w').close()

        # 12. set up the mining consensus 根据传入参数设置挖矿共识机制  pow_difficulty:0 为POS
        mining_consensus = 'PoW' if args['pow_difficulty'] else 'PoS'

    # create malicious local_enterprise identification database 创建一个SQLite数据库，记录本地企业节点的恶意行为识别日志
    conn = sqlite3.connect(f'{log_files_folder_path}/malicious_enterprise_identifying_log.db')
    conn_cursor = conn.cursor()
    '''
        日志中叫db，数据库中表叫malicious_local_enterprises_log，表中的字段如下所示
        enterprise_seq text 存储企业节点的序列号或者唯一标识符
        if_malicious integer 该节点是否为恶意节点
        correctly_identified_by text 哪些节点正确识别了节点的恶意状态
        incorrectly_identified_by text 哪些节点错误的将节点识别为恶意或者非恶意
        in_round integer 识别发生的轮数，追踪识别时间
        when_resyncing text 节点的识别或状态变化是否发生在区块链同步时
    '''
    conn_cursor.execute(
        """
        CREATE TABLE if not exists  malicious_local_enterprises_log (
        enterprise_seq text,
        if_malicious integer,
        correctly_identified_by text,
        incorrectly_identified_by text,
        in_round integer,
        when_resyncing text
    )""")

    target_accuracy = args['target_acc']  # 0.9 设定模型训练的目标精度

    # FedAnil+: Total Communication Cost (Bytes) 总通信成本，用于累加通信的总字节数目
    communication_bytes_sum = 0
    # FedAnil+: Total Computation Cost (Seconds) 总计算成本，累计所有节点的计算时间(以秒为单位)
    computation_sum = 0
    # FedAnil+: Total Accuracy (%) 总精度，用于记录模型的总精度，可以计算平均精度检查检查是否达到预期的target_accuracy
    total_accuracy = 0
    # FedAnil+ starts here
    for comm_round in range(latest_round_num + 1, args['max_num_comm'] + 1):  # 通信轮次循环 1:51
        communication_bytes_per_round = 0  # 初始化每轮的通信成本
        # create round specific log folder 创建轮次日志文件夹
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if os.path.exists(log_files_folder_path_comm_round):
            print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
            shutil.rmtree(log_files_folder_path_comm_round)  # 递归地删除目录树
        os.mkdir(log_files_folder_path_comm_round)
        # free cuda memory 释放cuda内存
        if dev == torch.device("cuda"):
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
        print(
            f"\n*****************************************Communication round {comm_round}*****************************************")
        # FedAnil+: Total Computation Cost 记录轮次开始时间
        comm_round_start_time = time.time()
        # (RE)ASSIGN ROLES 角色分配
        local_enterprises_to_assign = local_enterprises_needed  # 80
        miners_to_assign = miners_needed  # 10
        validators_to_assign = validators_needed  # 10
        local_enterprises_this_round = []
        miners_this_round = []
        validators_this_round = []
        random.shuffle(enterprises_list)  # 随机打乱
        # 奖励值从小到大排序,相当于奖励值小的本地企业，奖励值大的矿工
        enterprises_list.sort(key=lambda x: x.rewards, reverse=False)
        # 遍历企业节点，按需求分配角色，每个节点初始重置上一次的角色
        print('local_enterprises_to_assign:{},validator_to_assign:{},miners_to_assign:{}'.format(
            local_enterprises_to_assign, validators_to_assign, miners_to_assign))
        for enterprise in enterprises_list:
            enterprise.reset_last()  # lastprc = 0
            if local_enterprises_to_assign:
                enterprise.assign_local_enterprise_role()
                local_enterprises_to_assign -= 1
            elif validators_to_assign:
                enterprise.assign_validator_role()
                validators_to_assign -= 1
            elif miners_to_assign:
                enterprise.assign_miner_role()
                miners_to_assign -= 1
            else:
                enterprise.assign_role()
            if enterprise.return_role() == 'local_enterprise':
                local_enterprises_this_round.append(enterprise)
            elif enterprise.return_role() == 'miner':
                miners_this_round.append(enterprise)
            else:
                validators_this_round.append(enterprise)
            # determine if online at the beginning (essential for step 1 when local_enterprise needs to associate with an online enterprise)
            # 随机决定每个节点的在线状态,默认节点在线
            enterprise.online_switcher()

        # re-init round vars - in real distributed system, they could still fall behind in comm round, but here we assume they will all go into the next round together, thought enterprise may go offline somewhere in the previous round and their variables were not therefore reset
        # 重置参与某个轮次的矿工，企业，验证者
        for miner in miners_this_round:
            miner.miner_reset_vars_for_new_round()  # 重置矿工的状态变量
        for local_enterprise in local_enterprises_this_round:
            local_enterprise.local_enterprise_reset_vars_for_new_round()  # 重置地方企业的状态变量
        for validator in validators_this_round:
            validator.validator_reset_vars_for_new_round()  # 重置验证者的状态变量

        # DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for local_enterprise, this will affect the order of dataset portions to be trained)
        # 对当前轮次中的地方企业、矿工和验证者的列表进行随机打乱
        random.shuffle(local_enterprises_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)

        # selected miners 选择当前轮次的矿工，根据矿工的奖励值选择领导矿工
        selected_miners = miners_this_round
        stake_of_miners = []  # 存储每个选定矿工的奖励值reward
        for it in range(len(selected_miners)):
            stake_of_miners.append(selected_miners[it].return_stake())
        index_max = stake_of_miners.index(max(stake_of_miners))  # 奖励值最大的索引
        leader_miner = selected_miners[index_max]  # 将这个矿工作为本轮的领导者，奖励值最大的矿工
        # 随机选择一个数，范围从 80*0.8到80 64到80
        random_selection_num = random.randrange(int(len(local_enterprises_this_round) * 0.8),
                                                int(len(local_enterprises_this_round) * 1.0))

        ''' local_enterprises, validators and miners take turns to perform jobs '''
        # 选择前random_selection_num个企业 都是本地企业
        selected_local_enterprises_this_round = local_enterprises_this_round[0:random_selection_num]
        print(f"SELECTION : {random_selection_num} of {local_enterprises_needed} local enterprises in this round")
        print()
        print(
            ''' Step 1 - local_enterprises assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n''')

        # FedAnil+: Select Random numbers from enterprises
        ''' 
        DEBUGGING CODE
        在调试模式下，详细显示每轮次的企业、矿工、验证者的在线状态、区块链长度和对等节点列表
         '''
        if args['verbose']:

            print('*************This is a verbose debugging code**************')

            # show enterprises initial chain length and if online 遍历本轮选中的本地企业，打印每个企业的索引，角色，在线状态和区块链长度
            for enterprise in selected_local_enterprises_this_round:
                if enterprise.is_online():
                    print(f'{enterprise.return_idx()} {enterprise.return_role()} online - ', end='')
                else:
                    print(f'{enterprise.return_idx()} {enterprise.return_role()} offline - ', end='')
                # debug chain length
                print(f"chain length {enterprise.return_consortium_blockchain_object().return_chain_length()}")

            # show enterprise roles
            print(
                f"\nThere are {len(selected_local_enterprises_this_round)} local_enterprises, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
            print("\nLocal_enterprises this round are")
            for local_enterprise in selected_local_enterprises_this_round:
                print(
                    f"e_{local_enterprise.return_idx().split('_')[-1]} online - {local_enterprise.is_online()} with chain len {local_enterprise.return_consortium_blockchain_object().return_chain_length()}")
            print("\nMiners this round are")
            for miner in miners_this_round:
                print(
                    f"e_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_consortium_blockchain_object().return_chain_length()}")
            print("\nValidators this round are")
            for validator in validators_this_round:
                print(
                    f"e_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_consortium_blockchain_object().return_chain_length()}")
            print()
            # show peers with round number 打印出每个企业的对等peer列表，每个企业的索引，角色以及它的peer
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for enterprise_seq, enterprise in enterprises_in_network.enterprises_set.items():
                peers = enterprise.return_peers()
                print(f"e_{enterprise_seq.split('_')[-1]} - {enterprise.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"e_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print(f"+++++++++ Round {comm_round} ending Peer Lists +++++++++")

        print('*****************This verbose debugging code is done*****************')

        ''' DEBUGGING CODE ENDS '''

        # for local_enterprise_iter in range(len(local_enterprises_this_round)):
        # 本地企业与网络中的其他角色(验证者、矿工)交互，区块链获取全局模型，同步链，更新本地模型，与网络中的其他参与者建立联系
        for local_enterprise_iter in range(len(selected_local_enterprises_this_round)):
            # local_enterprise = local_enterprises_this_round[local_enterprise_iter]
            local_enterprise = selected_local_enterprises_this_round[local_enterprise_iter]
            # FedAnil+: fetch global model from consortium blockchain 本地企业从区块链对象中获取全局模型
            local_enterprise.fetch_global_model(local_enterprise.return_consortium_blockchain_object())
            # resync chain(block could be dropped due to fork from last round) 同步链 销毁存储在区块中的交易节省gpu内存
            if local_enterprise.resync_chain(mining_consensus):
                local_enterprise.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
            # FedAnil+: Total Communication Cost (Bytes): Transfer of Global Model Bytes from Server to Clients
            # 累加这一轮次中通信的总字节数
            communication_bytes_per_round += sys.getsizeof(local_enterprise.global_parameters)
            # local_enterprise (should) perform local update and associate
            # print(f"{local_enterprise.return_idx()} - local_enterprise {local_enterprise_iter+1}/{len(local_enterprises_this_round)} will associate with a validator and a miner, if online...")
            # 本地企业将尝试与validator和miner建立联系，如果他们在线的话
            print(
                f"-----------{local_enterprise.return_idx()} - local_enterprise {local_enterprise_iter + 1}/{len(selected_local_enterprises_this_round)} will associate with a validator and a miner, if online...------------")
            # local_enterprise associates with a miner to accept finally mined block
            if local_enterprise.online_switcher():
                # 本地企业尝试与矿工建立联系
                associated_miner = local_enterprise.associate_with_enterprise("miner")
                # 如果找到了矿工，矿工会将本地企业添加到其关联列表
                if associated_miner:
                    associated_miner.add_enterprise_to_association(local_enterprise)
                else:
                    print(f"Cannot find a qualified miner in {local_enterprise.return_idx()} peer list.")
            # local_enterprise associates with a validator to send local_enterprise transactions
            if local_enterprise.online_switcher():
                # 本地企业尝试与验证者建立联系
                associated_validator = local_enterprise.associate_with_enterprise("validator")
                if associated_validator:
                    # 如果找到了验证者，验证者会将本地企业添加到其关联列表
                    associated_validator.add_enterprise_to_association(local_enterprise)
                else:
                    print(f"Cannot find a qualified validator in {local_enterprise.return_idx()} peer list.")

        """执行第二步，验证者接受来自本地企业的更新，并将这些更新广播到它们各自到对等节点列表中其他到validator，调用本地企业到local_updates()"""
        print()
        print(
            ''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (local_enterprises local_updates() are called in this step.\n''')
        for validator_iter in range(len(validators_this_round)):  # len(validators_this_round) : 10
            validator = validators_this_round[validator_iter]
            # resync chain 同步区块链,默认为不同步区块链 None 销毁存储在区块中的交易节省gpu内存
            if validator.resync_chain(mining_consensus):  # mining_consensus默认为POS
                validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
            # 将当前验证者的全局模型参数的大小加到本轮通信的总子节数上，计算通信成本
            communication_bytes_per_round += sys.getsizeof(validator.global_parameters)
            # associate with a miner to send post validation transactions 与矿工建立关联
            if validator.online_switcher():
                associated_miner = validator.associate_with_enterprise("miner")
                if associated_miner:
                    associated_miner.add_enterprise_to_association(validator)
                else:
                    print(f"Cannot find a qualified miner in validator {validator.return_idx()} peer list.")
            # validator accepts local updates from its local_enterprises association 接受本地企业的更新
            # 获取与当前验证者关联的所有本地企业节点的列表，注意是set集合，是无序的
            associated_local_enterprises = list(validator.return_associated_local_enterprises())
            # 若没有本地企业与该验证者关联，则会打印一条信息并跳过该验证者，进入下一次循环，下一个验证者
            if not associated_local_enterprises:
                print(
                    f"No local_enterprises are associated with validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} for this communication round.")
                continue
            validator_link_speed = validator.return_link_speed()  # 70000.0
            # 如果验证者在线，根据网络链接速度接受来自关联的本地企业的更新，链接速度决定了数据传输的速度
            print(
                f"-------------------{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is accepting local_enterprises' updates with link speed {validator_link_speed} bytes/s, if online...--------------------")
            # records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
            # 这个字典用于记录每个epoch的传输延迟，以确定下一个epoch更新到达时间，每个键是验证者关联的每个企业的实例，初始值为None
            records_dict = dict.fromkeys(associated_local_enterprises, None)
            for local_enterprise, _ in records_dict.items():
                # 为每个与验证者关联的本地企业创建一个嵌套字典，这个嵌套字典存储与每个本地企业相关的详细信息
                records_dict[local_enterprise] = {}
            # used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
            transaction_arrival_queue = {}  # 初始化交易到达队列，记录从本地企业接收到的更新到达时间，用于后续的验证者广播和矿工接收的排序
            # local_enterprises local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
            if args['miner_acception_wait_time']:  # 矿工接受交易的默认时间窗口，单位为秒。0 表示无时间限制。默认为 0.0
                print(
                    f"miner wait time is specified as {args['miner_acception_wait_time']} seconds. let each local_enterprise do local_updates till time limit")
                for local_enterprise_iter in range(len(associated_local_enterprises)):
                    local_enterprise = associated_local_enterprises[local_enterprise_iter]
                    if not local_enterprise.return_idx() in validator.return_black_list():
                        # TODO here, also add print() for below miner's validators
                        print(
                            f'local_enterprise {local_enterprise_iter + 1}/{len(associated_local_enterprises)} of validator {validator.return_idx()} is doing local updates')
                        total_time_tracker = 0
                        update_iter = 1
                        local_enterprise_link_speed = local_enterprise.return_link_speed()
                        lower_link_speed = validator_link_speed if validator_link_speed < local_enterprise_link_speed else local_enterprise_link_speed
                        while total_time_tracker < validator.return_miner_acception_wait_time():
                            # simulate the situation that local_enterprise may go offline during model updates transmission to the validator, based on per transaction
                            if local_enterprise.online_switcher():
                                # local_enterprise local update
                                local_update_spent_time = local_enterprise.local_enterprise_local_update(rewards,
                                                                                                         log_files_folder_path_comm_round,
                                                                                                         comm_round)
                                unverified_transaction = local_enterprise.return_local_updates_and_signature(comm_round)
                                # size in bytes, usually around 35000 bytes per transaction
                                communication_bytes_per_round += local_enterprise.size_of_encoded_data
                                unverified_transactions_size = getsizeof(str(unverified_transaction))
                                transmission_delay = unverified_transactions_size / lower_link_speed
                                if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
                                    # last transaction sent passes the acception time window
                                    break
                                records_dict[local_enterprise][update_iter] = {}
                                records_dict[local_enterprise][update_iter][
                                    'local_update_time'] = local_update_spent_time
                                records_dict[local_enterprise][update_iter]['transmission_delay'] = transmission_delay
                                records_dict[local_enterprise][update_iter][
                                    'local_update_unverified_transaction'] = unverified_transaction
                                records_dict[local_enterprise][update_iter][
                                    'local_update_unverified_transaction_size'] = unverified_transactions_size
                                if update_iter == 1:
                                    total_time_tracker = local_update_spent_time + transmission_delay
                                else:
                                    total_time_tracker = total_time_tracker - \
                                                         records_dict[local_enterprise][update_iter - 1][
                                                             'transmission_delay'] + local_update_spent_time + transmission_delay
                                records_dict[local_enterprise][update_iter]['arrival_time'] = total_time_tracker
                                if validator.online_switcher():
                                    # accept this transaction only if the validator is online
                                    print(f"validator {validator.return_idx()} has accepted this transaction.")
                                    transaction_arrival_queue[total_time_tracker] = unverified_transaction
                                else:
                                    print(
                                        f"validator {validator.return_idx()} offline and unable to accept this transaction")
                            else:
                                # local_enterprise goes offline and skip updating for one transaction, wasted the time of one update and transmission
                                wasted_update_time, wasted_update_params = local_enterprise.waste_one_epoch_local_update_time(
                                    args['optimizer'])
                                wasted_update_params_size = getsizeof(str(wasted_update_params))
                                wasted_transmission_delay = wasted_update_params_size / lower_link_speed
                                if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
                                    # wasted transaction "arrival" passes the acception time window
                                    break
                                records_dict[local_enterprise][update_iter] = {}
                                records_dict[local_enterprise][update_iter]['transmission_delay'] = transmission_delay
                                if update_iter == 1:
                                    total_time_tracker = wasted_update_time + wasted_transmission_delay
                                    print(
                                        f"local_enterprise goes offline and wasted {total_time_tracker} seconds for a transaction")
                                else:
                                    total_time_tracker = total_time_tracker - \
                                                         records_dict[local_enterprise][update_iter - 1][
                                                             'transmission_delay'] + wasted_update_time + wasted_transmission_delay
                            update_iter += 1
            else:
                print()
                print('The parameter miner_acception_wait_time is zero')
                # 没有指定等待时间，每个关联的local_enterprise执行指定数量的本地epoch
                # did not specify wait time. every associated local_enterprise perform specified number of local epochs
                # 处理validator和local_enterprise之间的交互，模拟本地企业进行本地更新并将这些更新发送给验证者的过程
                for local_enterprise_iter in range(len(associated_local_enterprises)):
                    # 遍历与当前验证者关联的所有本地企业
                    local_enterprise = associated_local_enterprises[local_enterprise_iter]
                    if not local_enterprise.return_idx() in validator.return_black_list():
                        # 当前本地企业不在验证者的黑名单中，说明这个本地企业是可信的,打印信息，表明当前本地企业正在执行本地更新
                        print(
                            f'local_enterprise {local_enterprise.return_idx()}  {local_enterprise_iter + 1}/{len(associated_local_enterprises)} of validator {validator.return_idx()} is doing local updates')
                        if local_enterprise.online_switcher():
                            # 本地企业在线，继续本地更新，执行更新并返回更新所花费的时间
                            local_update_spent_time = local_enterprise.local_enterprise_local_update(rewards,
                                                                                                     log_files_folder_path_comm_round,
                                                                                                     comm_round,
                                                                                                     local_epochs=args[
                                                                                                         'default_local_epochs'])
                            local_enterprise_link_speed = local_enterprise.return_link_speed()
                            lower_link_speed = validator_link_speed if validator_link_speed < local_enterprise_link_speed else local_enterprise_link_speed
                            unverified_transaction = local_enterprise.return_local_updates_and_signature(comm_round)
                            unverified_transactions_size = getsizeof(str(unverified_transaction))
                            transmission_delay = unverified_transactions_size / lower_link_speed
                            if validator.online_switcher():
                                transaction_arrival_queue[
                                    local_update_spent_time + transmission_delay] = unverified_transaction
                                print(f"validator {validator.return_idx()} has accepted this transaction.")
                            else:
                                print(
                                    f"validator {validator.return_idx()} offline and unable to accept this transaction")
                        else:
                            # 本地企业不在线，打印信息
                            print(
                                f"local_enterprise {local_enterprise.return_idx()} offline and unable do local updates")
                    else:
                        # 本地企业在验证者的黑名单中，这个企业不可信，打印信息
                        print(
                            f"local_enterprise {local_enterprise.return_idx()} in validator {validator.return_idx()}'s black list. This local_enterprise's transactions won't be accpeted.")
            validator.set_unordered_arrival_time_accepted_local_enterprise_transactions(transaction_arrival_queue)
            # in case validator off line for accepting broadcasted transactions but can later back online to validate the transactions itself receives
            validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))

            # broadcast to other validators
            if transaction_arrival_queue:
                validator.validator_broadcast_local_enterprise_transactions()
            else:
                print(
                    "No transactions have been received by this validator, probably due to local_enterprises and/or validators offline or timeout while doing local updates or transmitting updates, or all local_enterprises are in validator's black list.")

        print(
            ''' Step 2.5 - with the broadcasted local_enterprises transactions, validators decide the final transaction arrival order \n''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            accepted_broadcasted_validator_transactions = validator.return_accepted_broadcasted_local_enterprise_transactions()
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is calculating the final transactions arrival order by combining the direct local_enterprise transactions received and received broadcasted transactions...")
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_validator_transactions:
                # calculate broadcasted transactions arrival time
                self_validator_link_speed = validator.return_link_speed()
                for broadcasting_validator_record in accepted_broadcasted_validator_transactions:
                    broadcasting_validator_link_speed = broadcasting_validator_record['source_validator_link_speed']
                    lower_link_speed = self_validator_link_speed if self_validator_link_speed < broadcasting_validator_link_speed else broadcasting_validator_link_speed
                    for arrival_time_at_broadcasting_validator, broadcasted_transaction in \
                            broadcasting_validator_record['broadcasted_transactions'].items():
                        transmission_delay = getsizeof(str(broadcasted_transaction)) / lower_link_speed
                        accepted_broadcasted_transactions_arrival_queue[
                            transmission_delay + arrival_time_at_broadcasting_validator] = broadcasted_transaction
            else:
                print(
                    f"validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} did not receive any broadcasted local_enterprise transaction this round.")
            # mix the boardcasted transactions with the direct accepted transactions
            final_transactions_arrival_queue = sorted(
                {**validator.return_unordered_arrival_time_accepted_local_enterprise_transactions(),
                 **accepted_broadcasted_transactions_arrival_queue}.items())
            validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")

        print(
            ''' Step 3 - validators do self and cross-validation(validate local updates from local_enterprises) by the order of transaction arrival time.\n''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
            if final_transactions_arrival_queue:
                # validator asynchronously does one epoch of update and validate on its own test set
                local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(
                    args['optimizer'])
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is validating received local_enterprise transactions...")
                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    if validator.online_switcher():
                        # validation won't begin until validator locally done one epoch of update and validation(local_enterprise transactions will be queued)
                        if arrival_time < local_validation_time:
                            arrival_time = local_validation_time
                        validation_time, post_validation_unconfirmmed_transaction = validator.validate_local_enterprise_transaction(
                            unconfirmmed_transaction, rewards, log_files_folder_path, comm_round,
                            args['malicious_validator_on'])
                        if validation_time:
                            validator.add_post_validation_transaction_to_queue((arrival_time + validation_time,
                                                                                validator.return_link_speed(),
                                                                                post_validation_unconfirmmed_transaction))
                            print(
                                f"A validation process has been done for the transaction from local_enterprise {post_validation_unconfirmmed_transaction['local_enterprise_enterprise_idx']} by validator {validator.return_idx()}")
                    else:
                        print(
                            f"A validation process is skipped for the transaction from local_enterprise {post_validation_unconfirmmed_transaction['local_enterprise_enterprise_idx']} by validator {validator.return_idx()} due to validator offline.")
            else:
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} did not receive any transaction from local_enterprise or validator in this round.")

        print(
            ''' Step 4 - validators send post validation transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists\n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            # resync chain
            if miner.resync_chain(mining_consensus):
                miner.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)

            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} accepting validators' post-validation transactions...")
            associated_validators = list(miner.return_associated_validators())
            if not associated_validators:
                print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
                continue
            self_miner_link_speed = miner.return_link_speed()
            validator_transactions_arrival_queue = {}
            for validator_iter in range(len(associated_validators)):
                validator = associated_validators[validator_iter]
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(associated_validators)} of miner {miner.return_idx()} is sending signature verified transaction...")
                post_validation_transactions_by_validator = validator.return_post_validation_transactions_queue()
                post_validation_unconfirmmed_transaction_iter = 1
                for (validator_sending_time, source_validator_link_spped,
                     post_validation_unconfirmmed_transaction) in post_validation_transactions_by_validator:
                    if validator.online_switcher() and miner.online_switcher():
                        lower_link_speed = self_miner_link_speed if self_miner_link_speed < source_validator_link_spped else source_validator_link_spped
                        transmission_delay = getsizeof(str(post_validation_unconfirmmed_transaction)) / lower_link_speed
                        validator_transactions_arrival_queue[
                            validator_sending_time + transmission_delay] = post_validation_unconfirmmed_transaction
                        print(
                            f"miner {miner.return_idx()} has accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()}")
                    else:
                        print(
                            f"miner {miner.return_idx()} has not accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()} due to one of enterprises or both offline.")
                    post_validation_unconfirmmed_transaction_iter += 1
            miner.set_unordered_arrival_time_accepted_validator_transactions(validator_transactions_arrival_queue)
            miner.miner_broadcast_validator_transactions()

        print(
            ''' Step 4.5 - with the broadcasted validator transactions, miners decide the final transaction arrival order\n ''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            accepted_broadcasted_validator_transactions = miner.return_accepted_broadcasted_validator_transactions()
            self_miner_link_speed = miner.return_link_speed()
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} calculating the final transactions arrival order by combining the direct local_enterprise transactions received and received broadcasted transactions...")
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_validator_transactions:
                # calculate broadcasted transactions arrival time
                for broadcasting_miner_record in accepted_broadcasted_validator_transactions:
                    broadcasting_miner_link_speed = broadcasting_miner_record['source_enterprise_link_speed']
                    lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed
                    for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record[
                        'broadcasted_transactions'].items():
                        transmission_delay = getsizeof(str(broadcasted_transaction)) / lower_link_speed
                        accepted_broadcasted_transactions_arrival_queue[
                            transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
            else:
                print(
                    f"miner {miner.return_idx()} {miner_iter + 1}/{len(miners_this_round)} did not receive any broadcasted validator transaction this round.")
            # mix the boardcasted transactions with the direct accepted transactions
            final_transactions_arrival_queue = sorted(
                {**miner.return_unordered_arrival_time_accepted_validator_transactions(),
                 **accepted_broadcasted_transactions_arrival_queue}.items())
            miner.set_candidate_transactions_for_final_mining_queue(final_transactions_arrival_queue)
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")

        print(
            ''' Step 5 - miners do self and cross-verification (verify validators' signature) by the order of transaction arrival time and record the transactions in the candidate block according to the limit size. Also mine and propagate the block.\n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            final_transactions_arrival_queue = miner.return_final_candidate_transactions_mining_queue()
            valid_validator_sig_candidate_transacitons = []
            invalid_validator_sig_candidate_transacitons = []
            begin_mining_time = 0
            if final_transactions_arrival_queue:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is verifying received validator transactions...")
                time_limit = miner.return_miner_acception_wait_time()
                size_limit = miner.return_miner_accepted_transactions_size_limit()
                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    new_begining_mining_time = 0
                    if miner.online_switcher():
                        if time_limit:
                            if arrival_time > time_limit:
                                break
                        if size_limit:
                            if getsizeof(
                                    str(valid_validator_sig_candidate_transacitons + invalid_validator_sig_candidate_transacitons)) > size_limit:
                                break
                        # verify validator signature of this transaction
                        verification_time, is_validator_sig_valid = miner.verify_validator_transaction(
                            unconfirmmed_transaction)
                        if verification_time:
                            if is_validator_sig_valid:
                                validator_info_this_tx = {
                                    'validator': unconfirmmed_transaction['validation_done_by'],
                                    'validation_rewards': unconfirmmed_transaction['validation_rewards'],
                                    'validation_time': unconfirmmed_transaction['validation_time'],
                                    'validator_rsa_pub_key': unconfirmmed_transaction['validator_rsa_pub_key'],
                                    'validator_signature': unconfirmmed_transaction['validator_signature'],
                                    'update_direction': unconfirmmed_transaction['update_direction'],
                                    'miner_enterprise_idx': miner.return_idx(),
                                    'miner_verification_time': verification_time,
                                    'miner_rewards_for_this_tx': rewards}
                                # validator's transaction signature valid
                                found_same_local_enterprise_transaction = False
                                for valid_validator_sig_candidate_transaciton in valid_validator_sig_candidate_transacitons:
                                    if valid_validator_sig_candidate_transaciton['local_enterprise_signature'] == \
                                            unconfirmmed_transaction['local_enterprise_signature']:
                                        found_same_local_enterprise_transaction = True
                                        break
                                if not found_same_local_enterprise_transaction:
                                    valid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                    del valid_validator_sig_candidate_transaciton['validation_done_by']
                                    del valid_validator_sig_candidate_transaciton['validation_rewards']
                                    del valid_validator_sig_candidate_transaciton['update_direction']
                                    del valid_validator_sig_candidate_transaciton['validation_time']
                                    del valid_validator_sig_candidate_transaciton['validator_rsa_pub_key']
                                    del valid_validator_sig_candidate_transaciton['validator_signature']
                                    valid_validator_sig_candidate_transaciton['positive_direction_validators'] = []
                                    valid_validator_sig_candidate_transaciton['negative_direction_validators'] = []
                                    valid_validator_sig_candidate_transacitons.append(
                                        valid_validator_sig_candidate_transaciton)
                                if unconfirmmed_transaction['update_direction']:
                                    valid_validator_sig_candidate_transaciton['positive_direction_validators'].append(
                                        validator_info_this_tx)
                                else:
                                    valid_validator_sig_candidate_transaciton['negative_direction_validators'].append(
                                        validator_info_this_tx)
                                transaction_to_sign = valid_validator_sig_candidate_transaciton
                            else:
                                # validator's transaction signature invalid
                                invalid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                invalid_validator_sig_candidate_transaciton[
                                    'miner_verification_time'] = verification_time
                                invalid_validator_sig_candidate_transaciton['miner_rewards_for_this_tx'] = rewards
                                invalid_validator_sig_candidate_transacitons.append(
                                    invalid_validator_sig_candidate_transaciton)
                                transaction_to_sign = invalid_validator_sig_candidate_transaciton
                            # (re)sign this candidate transaction
                            signing_time = miner.sign_candidate_transaction(transaction_to_sign)
                            new_begining_mining_time = arrival_time + verification_time + signing_time
                    else:
                        print(
                            f"A verification process is skipped for the transaction from validator {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline.")
                        new_begining_mining_time = arrival_time
                    begin_mining_time = new_begining_mining_time if new_begining_mining_time > begin_mining_time else begin_mining_time
                # FedAnil+: add global params to transaction of block
                transactions_to_record_in_block = {}
                transactions_to_record_in_block[
                    'valid_validator_sig_transacitons'] = valid_validator_sig_candidate_transacitons
                transactions_to_record_in_block[
                    'invalid_validator_sig_transacitons'] = invalid_validator_sig_candidate_transacitons
                transactions_to_record_in_block['global_update_params'] = miner.return_global_parametesrs()
                # put transactions into candidate block and begin mining
                # block index starts from 1
                start_time_point = time.time()
                candidate_block = Block(idx=miner.return_consortium_blockchain_object().return_chain_length() + 1,
                                        transactions=transactions_to_record_in_block,
                                        miner_rsa_pub_key=miner.return_rsa_pub_key())
                # mine the block
                miner_computation_power = miner.return_computation_power()
                if not miner_computation_power:
                    block_generation_time_spent = float('inf')
                    miner.set_block_generation_time_point(float('inf'))
                    print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
                    continue
                recorded_transactions = candidate_block.return_transactions()
                if recorded_transactions['valid_validator_sig_transacitons'] or recorded_transactions[
                    'valid_validator_sig_transacitons']:
                    print(f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} mining the block...")
                    # return the last block and add previous hash
                    last_block = miner.return_consortium_blockchain_object().return_last_block()
                    if last_block is None:
                        # will mine the genesis block
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.mine_block(candidate_block, rewards)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline while propagating its block
                if miner.online_switcher():
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
                    # record mining time
                    block_generation_time_spent = (time.time() - start_time_point) / miner_computation_power
                    miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
                    print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
                    # immediately propagate the block
                    miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
                else:
                    print(
                        f"Unfortunately, {miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated.")
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} did not receive any transaction from validator or miner in this round.")

        print(
            ''' Step 6 - miners decide if adding a propagated block or its own mined block as the legitimate block, and request its associated enterprises to download this block''')
        forking_happened = False
        # comm_round_block_gen_time regarded as the time point when the winning miner mines its block, calculated from the beginning of the round. If there is forking in PoW or rewards info out of sync in PoS, this time is the avg time point of all the appended time by any enterprise
        comm_round_block_gen_time = []
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
            # add self mined block to the processing queue and sort by time
            this_miner_mined_block = miner.return_mined_block()
            leader_miner = miner
            if this_miner_mined_block:
                unordered_propagated_block_processing_queue[
                    miner.return_block_generation_time_point()] = this_miner_mined_block
            ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
            if ordered_all_blocks_processing_queue:
                if mining_consensus == 'PoW':
                    print("\nselect winning block based on PoW")

                    # abort mining if propagated block is received
                    print(
                        f"{leader_miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block...")
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        verified_block, verification_time = leader_miner.verify_block(block_to_verify,
                                                                                      block_to_verify.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == leader_miner.return_idx():
                                print(f"Miner {leader_miner.return_idx()} is adding its own mined block.")
                            else:
                                print(
                                    f"Miner {leader_miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
                            if leader_miner.online_switcher():
                                # FedAnil+: upload global model to consortium blockchain
                                leader_miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {leader_miner.return_idx()} goes offline while adding this block to its chain.")
                            if leader_miner.return_the_added_block():
                                # requesting enterprises in its associations to download this block
                                leader_miner.request_to_download(verified_block, block_arrival_time + verification_time)
                                break
                else:
                    # PoS
                    candidate_PoS_blocks = {}
                    print("select winning block based on PoS")
                    # filter the ordered_all_blocks_processing_queue to contain only the blocks within time limit
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        if block_arrival_time < args['miner_pos_propagated_block_wait_time']:
                            candidate_PoS_blocks[enterprises_in_network.enterprises_set[
                                block_to_verify.return_mined_by()].return_stake()] = block_to_verify
                    high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
                    # for PoS, requests every enterprise in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
                    for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
                        verified_block, verification_time = miner.verify_block(PoS_candidate_block,
                                                                               PoS_candidate_block.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == leader_miner.return_idx():
                                print(
                                    f"Miner {leader_miner.return_idx()} with stake {stake} is adding its own mined block.")
                            else:
                                print(
                                    f"Miner {leader_miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
                            if leader_miner.online_switcher():
                                # FedAnil+: upload global model to consortium blockchain
                                leader_miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {leader_miner.return_idx()} goes offline while adding this block to its chain.")
                            if leader_miner.return_the_added_block():
                                # requesting enterprises in its associations to download this block
                                leader_miner.request_to_download(verified_block, block_arrival_time + verification_time)
                                break
                leader_miner.add_to_round_end_time(block_arrival_time + verification_time)
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet.")
        # CHECK FOR FORKING
        added_blocks_miner_set = set()
        for enterprise in enterprises_list:
            the_added_block = enterprise.return_the_added_block()
            if the_added_block:
                print(
                    f"{enterprise.return_role()} {enterprise.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
                added_blocks_miner_set.add(the_added_block.return_mined_by())
                block_generation_time_point = enterprises_in_network.enterprises_set[
                    the_added_block.return_mined_by()].return_block_generation_time_point()
                # commented, as we just want to plot the legitimate block gen time, and the wait time is to avoid forking,Also the logic is wrong. Should track the time to the slowest local_enterprise after its global model update
                # if mining_consensus == 'PoS':
                # 	if args['miner_pos_propagated_block_wait_time'] != float("inf"):
                # 		block_generation_time_point += args['miner_pos_propagated_block_wait_time']
                comm_round_block_gen_time.append(block_generation_time_point)
        if len(added_blocks_miner_set) > 1:
            print("WARNING: a forking event just happened!")
            forking_happened = True
            with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file:
                file.write(f"Forking in round {comm_round}\n")
        else:
            print("No forking event happened.")

        print(
            ''' Step 6 last step - process the added block - 1.collect usable updated params\n 2.malicious enterprises identification\n 3.get rewards\n 4.do local udpates\n This code block is skipped if no valid block was generated in this round''')
        all_enterprises_round_ends_time = []
        for enterprise in enterprises_list:
            if enterprise.return_the_added_block() and enterprise.online_switcher():
                # collect usable updated params, malicious enterprises identification, get rewards and do local udpates
                processing_time = enterprise.process_block(enterprise.return_the_added_block(), log_files_folder_path,
                                                           conn, conn_cursor)
                enterprise.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
                enterprise.add_to_round_end_time(processing_time)
                all_enterprises_round_ends_time.append(enterprise.return_round_end_time())
        # FedAnil+: Total Accuracy (%)
        print(''' Logging Accuracies by Enterprises ''')
        for enterprise in enterprises_list:
            accuracy_this_round = enterprise.validate_model_weights()
            if total_accuracy < accuracy_this_round:
                total_accuracy = accuracy_this_round
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if enterprise.return_is_malicious() else "B"
                file.write(
                    f"{enterprise.return_idx()} {enterprise.return_role()} {is_malicious_node}: {accuracy_this_round}\n")

        # FedAnil+: Total Computation Cost (Seconds)
        communication_bytes_sum += communication_bytes_per_round
        # logging time, mining_consensus and forking
        # get the slowest enterprise end time
        # # FedAnil+: Total Computation Cost (Seconds)
        comm_round_spent_time = time.time() - comm_round_start_time
        with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
            # corner case when all miners in this round are malicious enterprises so their blocks are rejected
            try:
                comm_round_block_gen_time = max(comm_round_block_gen_time)
                file.write(f"comm_round_block_gen_time: {comm_round_block_gen_time}\n")
                file.write(f"communication overhead: {communication_bytes_per_round} bytes\n")
            except:
                no_block_msg = "No valid block has been generated this round."
                print(no_block_msg)
                file.write(f"comm_round_block_gen_time: {no_block_msg}\n")
                with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a+') as file2:
                    # TODO this may be caused by "no transaction to mine" for the miner. Forgot to check for block miner's maliciousness in request_to_downlaod()
                    file2.write(f"No valid block in round {comm_round}\n")
            try:
                slowest_round_ends_time = max(all_enterprises_round_ends_time)
                file.write(f"slowest_enterprise_round_ends_time: {slowest_round_ends_time}\n")
            except:
                # corner case when all transactions are rejected by miners
                file.write("slowest_enterprise_round_ends_time: No valid block has been generated this round.\n")
                with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a+') as file2:
                    no_valid_block_msg = f"No valid block in round {comm_round}\n"
                    if len(file2.readlines()) > 0:
                        if file2.readlines()[-1] != no_valid_block_msg:
                            file2.write(no_valid_block_msg)
            file.write(f"mining_consensus: {mining_consensus} {args['pow_difficulty']}\n")
            file.write(f"forking_happened: {forking_happened}\n")
            file.write(f"comm_round_spent_time_on_this_machine: {comm_round_spent_time}\n")
            # FedAnil+: Total Computation Cost (Second)
            computation_sum += comm_round_spent_time
        conn.commit()

        # if no forking, log the block miner
        if not forking_happened:
            legitimate_block = None
            for enterprise in enterprises_list:
                legitimate_block = enterprise.return_the_added_block()
                if legitimate_block is not None:
                    # skip the enterprise who's been identified malicious and cannot get a block from miners
                    break
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                if legitimate_block is None:
                    file.write("block_mined_by: no valid block generated this round\n")
                else:
                    block_mined_by = legitimate_block.return_mined_by()
                    is_malicious_node = "M" if enterprises_in_network.enterprises_set[
                        block_mined_by].return_is_malicious() else "B"
                    file.write(f"block_mined_by: {block_mined_by} {is_malicious_node}\n")
        else:
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                file.write(f"block_mined_by: Forking happened\n")

        with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
            file.write(f"Malicious Enterprises:\n")
            for enterprise in enterprises_list:
                if enterprise.return_is_malicious():
                    file.write(f"{enterprise.return_idx()}, ")
        print(''' Logging Stake by Enterprises ''')
        for enterprise in enterprises_list:
            accuracy_this_round = enterprise.validate_model_weights()
            with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if enterprise.return_is_malicious() else "B"
                file.write(
                    f"{enterprise.return_idx()} {enterprise.return_role()} {is_malicious_node}: {enterprise.return_stake()}\n")

        # a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
        if args['destroy_tx_in_block']:
            for enterprise in enterprises_list:
                last_block = enterprise.return_consortium_blockchain_object().return_last_block()
                if last_block:
                    last_block.free_tx()

        # save network_snapshot if reaches save frequency
        if args['save_network_snapshots'] and (comm_round == 1 or comm_round % args['save_freq'] == 0):
            if args['save_most_recent']:
                paths = sorted(Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime)
                if len(paths) > args['save_most_recent']:
                    for _ in range(len(paths) - args['save_most_recent']):
                        open(paths[_], 'w').close()
                        os.remove(paths[_])
            snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
            print(f"Saving network snapshot to {snapshot_file_path}")
            pickle.dump(enterprises_in_network, open(snapshot_file_path, "wb"))
        # FedAnil+: if accuracy reach more than target accuracy the iteration finished
        if total_accuracy >= target_accuracy:
            break

    with open(f'{log_files_folder_path}/Output.txt', 'w') as f:
        # FedAnil+: Total Computation Cost (Seconds)
        f.write(f"Total Computation Cost (Seconds): {computation_sum} \n")
        # FedAnil+: Total Communication Cost (Bytes)
        f.write(f"Total Communication Cost (Bytes): {communication_bytes_sum} \n")
        # FedAnil+: Total Accuracy (%)
        f.write(f"Total Accuracy (%): {total_accuracy * 100} \n")
