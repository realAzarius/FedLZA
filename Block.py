# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
import copy
# sha256哈希函数，生成区块的哈希值
from hashlib import sha256
import sys

'''
区块链中区块的基本功能
存储交易，计算哈希，验证签名，处理挖掘奖励等

'''


class Block:
    def __init__(self, idx, previous_block_hash=None, transactions=None, nonce=0, miner_rsa_pub_key=None, mined_by=None,
                 mining_rewards=None, pow_proof=None, signature=None):
        """
        idx:区块的索引
        previous_block_hash：前一个区块的哈希值，用于链接区块
        transactions：该区块包含的交易列表。
        nonce：一个随机数，用于工作量证明（PoW）算法。
        miner_rsa_pub_key：矿工的RSA公钥，用于验证矿工身份。
        mined_by：指示哪个矿工挖掘了这个区块。
        mining_rewards：矿工获得的奖励。
        pow_proof：工作量证明的结果。
        signature：区块的签名，用于验证区块的完整性。
        """
        self._idx = idx
        self._previous_block_hash = previous_block_hash
        self._transactions = transactions
        self._nonce = nonce
        # miner specific
        self._miner_rsa_pub_key = miner_rsa_pub_key
        self._mined_by = mined_by
        self._mining_rewards = mining_rewards
        # validator specific
        # self._is_validator_block = is_validator_block
        # the hash of the current block, calculated by compute_hash
        self._pow_proof = pow_proof
        self._signature = signature

    # compute_hash() also used to return value for block verification
    # if False by default, used for pow and verification, in which pow_proof has to be None, because at this moment -
    # pow - block hash is None, so does not affect much
    # verification - the block already has its hash
    # if hash_entire_block == True -> used in set_previous_block_hash, where we need to hash the whole previous block
    def compute_hash(self, hash_entire_block=False):
        """
        定义一个计算区块哈希值的方法。根据 hash_entire_block 参数的值决定是否计算整个区块的哈希。
        """
        block_content = copy.deepcopy(self.__dict__)
        if not hash_entire_block:
            block_content['_pow_proof'] = None
            block_content['_signature'] = None
            block_content['_mining_rewards'] = None
        # need sort keys to preserve order of key value pairs
        # 将区块内容转换为字符串，排序后计算SHA-256哈希，并返回哈希值。
        return sha256(str(sorted(block_content.items())).encode('utf-8')).hexdigest()

    def remove_signature_for_verification(self):
        """
        在验证区块时移除区块的签名
        """
        self._signature = None

    def set_pow_proof(self, the_hash):
        """
        设置工作量证明的哈希值
        """
        self._pow_proof = the_hash

    def nonce_increment(self):
        """
        增加 nonce 的值，通常用于挖矿过程中
        """
        self._nonce += 1

    # returners of the private attributes

    def return_previous_block_hash(self):
        """
        区块的前一个区块哈希
        """
        return self._previous_block_hash

    def return_block_idx(self):
        """
        区块索引
        """
        return self._idx

    def return_pow_proof(self):
        """
        工作量证明
        """
        return self._pow_proof

    def return_miner_rsa_pub_key(self):
        """
        矿工公钥
        """
        return self._miner_rsa_pub_key

    ''' Miner Specific '''

    def set_previous_block_hash(self, hash_to_set):
        """
        设置当前区块的前一个区块哈希
        """
        self._previous_block_hash = hash_to_set

    def add_verified_transaction(self, transaction):
        """
        向区块中添加已验证的交易
        """
        # after verified in cross_verification()
        # transactions can be both local_enterprises' or validators' transactions
        self._transactions.append(transaction)

    def set_nonce(self, nonce):
        """
        设置矿工的nonce
        """
        self._nonce = nonce

    def set_mined_by(self, mined_by):
        """
        设置挖掘者
        """
        self._mined_by = mined_by

    def return_mined_by(self):
        """
        返回挖掘者
        """
        return self._mined_by

    def set_signature(self, signature):
        """
        设置签名
        """
        # signed by mined_by node
        self._signature = signature

    def return_signature(self):
        """
        返回区块的签名
        """
        return self._signature

    def set_mining_rewards(self, mining_rewards):
        """
        设置挖掘奖励，矿工获得的奖励
        """
        self._mining_rewards = mining_rewards

    def return_mining_rewards(self):
        """
        返回挖掘奖励
        """
        return self._mining_rewards

    def return_transactions(self):
        """
        返回交易列表
        """
        return self._transactions

    # a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
    def free_tx(self):
        """
        用于释放区块中存储的交易，以节省内存。这个方法尝试删除交易列表，如果失败则捕获异常。
        """
        try:
            del self._transactions
        except:
            pass

