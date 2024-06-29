from dataclasses import dataclass
from typing import Any
import random 

import torch
from torch.utils.data import Sampler

# Randomly picks an index from an array of indices
class PoolSampler():
	def __init__( self, pool = [], keep_all = False ):
		self.length = len(pool)
		self.global_pool = pool if keep_all else None
		self.global_indices = [ i for i in range(self.length) ]
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

	def __len__(self):
		return self.length # len(self.current_pool)

	def __iter__(self):
		while len(self.current_pool) > 0:
			yield self.sample()

	def __call__(self, *args, **kwargs):
		return self.sample(*args, **kwargs)

	def get_state(self):
		return { "length": self.length, "global_pool": self.global_pool, "global_indices": self.global_indices, "current_pool": self.current_pool }
	
	def set_state(self, state):
		self.length = state["length"]
		self.global_pool = state["global_pool"]
		self.global_indices = state["global_indices"]
		self.current_pool = state["current_pool"]

# "Samples" through a fixed sequence from 0 to length
# Necessary for our "shuffle+sort by duration+interleave" sampling method
# Allows saving and loading state
class OrderedSampler(Sampler):
	def __init__( self, length ):
		self.position = 0
		self.length = length

	def __len__(self):
		return self.length

	def __iter__(self):
		if self.position >= self.length:
			self.position = 0

		while self.position < self.length:
			yield self.position
			self.position += 1

	def get_state(self):
		return { "position": self.position, "length": self.length }
	
	def set_state(self, state):
		self.position = state["position"]
		self.length = state["length"]

# Like the above, but will batch based on token count
class BatchedOrderedSampler(Sampler):
	def __init__( self, buckets, max_duration=0, max_batch_size=0 ):
		self.position = 0
		self.batches = []

		assert max_duration != 0 and max_batch_size != 0, "max_duration and max_batch_size cannot both be 0"

		current_batch = []
		current_size = 0
		current_index = 0
		for key, bucket in buckets.items():
			for path, duration in bucket:
				# flush
				should_flush = False
				if max_duration > 0 and current_size + duration > max_duration:
					should_flush = True
				elif max_batch_size > 0 and len(current_batch) >= max_batch_size:
					should_flush = True

				if should_flush and len(current_batch) > 0:
					self.batches.append( current_batch )
					current_batch = []
					current_size = 0
				
				current_batch.append( current_index )
				current_index += 1
				current_size += duration

	def __len__(self):
		return len(self.batches)

	def __iter__(self):
		if self.position >= len(self.batches):
			self.position = 0

		while self.position < len(self.batches):
			yield self.batches[self.position]
			self.position += 1

	def get_state(self):
		return { "position": self.position, "batches": self.batches }
	
	def set_state(self, state):
		self.position = state["position"]
		self.batches = state["batches"]

# Randomly samples indices from a given sequence from 0 to length
# Allows saving and loading state
class RandomSampler(Sampler):
	def __init__( self, length ):
		self.position = 0
		self.length = length

		self.generator = torch.Generator()
		self.perm = torch.randperm(self.length, generator=self.generator)

	def __len__(self):
		return self.length

	def __iter__(self):
		if self.position >= self.length:
			self.position = 0
			self.perm = torch.randperm(self.length, generator=self.generator)

		while self.position < self.length:
			yield self.perm[self.position]
			self.position += 1

	def get_state(self):
		return { "position": self.position, "length": self.length, "perm": self.perm, "generator": self.generator.get_state() }
	
	def set_state(self, state):
		self.position = state["position"]
		self.length = state["length"]
		self.perm = state["perm"]
		self.generator.set_state(state["generator"])