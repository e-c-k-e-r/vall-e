from dataclasses import dataclass
from typing import Any
import random 

@dataclass
class Sampler():
	def __init__( self, pool = [], keep_all = False ):
		self.global_pool = pool if keep_all else None
		self.global_indices = [ i for i in range(len(pool)) ]
		self.reset()

	def reset(self):
		self.current_pool = [ i for i in self.global_indices ]

	def sample(self, pool = None):
		if pool is None:
			pool = self.global_pool
		# check if we need to reset 
		index = random.choice( self.current_pool )
		# remove from pool
		self.current_pool.remove(index)
		# reset if needed
		if len(self.current_pool) == 0:
			self.reset()
		# map indices to our real values
		return pool[index] if pool is not None else index

	def __call__(self, *args, **kwargs):
		return self.sample(*args, **kwargs)