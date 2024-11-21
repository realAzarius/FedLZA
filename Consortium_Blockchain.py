from Block import Block  # 区块链中的一个区块
import copy

"""

提供了一个基本的框架来模拟区块链的行为，包括添加区块、管理临时数据和区块，以及获取区块链的当前状态。

"""

class Consortium_Blockchain:
	"""
	联盟区块链
	"""

	def __init__(self):
		self.chain = []
		self.tmpchain = []
		self.tmpdata = []

	def return_chain_structure(self):
		"""
		返回区块链的结构
		"""
		return self.chain

	def return_chain_length(self):
		"""
		返回区块链的长度
		"""
		return len(self.chain)

	def return_last_block(self):
		"""
		返回区块链中的最后一个区块。如果链为空（即没有区块），则返回None。
		"""
		if len(self.chain) > 0:
			return self.chain[-1]
		else:
			# blockchain doesn't even have its genesis block
			return None

	def return_last_block_pow_proof(self):
		"""
		返回最后一个区块的工作量证明POW
		"""
		if len(self.chain) > 0:
			return self.return_last_block().compute_hash(hash_entire_block=True)
		else:
			return None

	def replace_chain(self, chain):
		"""
		用传入的链替换当前的区块链
		"""
		self.chain = copy.copy(chain)

	def append_block(self, block):
		"""
		将一个新的区块添加到区块链的末尾
		"""
		self.chain.append(copy.copy(block))

	def new_local_block(self, block, cdata = None):
		"""
		创建一个新的本地区块，并将其添加到self.tmpchain列表中
		"""
		self.tmpchain.append(block)
		if cdata is not None:
			self.tmpdata.append(cdata)

	def return_local_chain(self):
		"""
		返回临时链
		"""
		return self.tmpchain
	
	def return_last_cdata(self):
		"""
		返回self.tmpdata列表中的最后一个数据项
		"""
		if len(self.tmpdata) > 0:
			return self.tmpdata[-1]
		else:
			return None