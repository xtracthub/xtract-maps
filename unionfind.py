from collections import defaultdict
import numpy as np


class UnionFind:
    '''
    Helper class to implement union-find / disjoint sets algorithm efficiently.

    The parent array stores the indices of the parent of the corresponding node.
    The size array stores the size of an up-tree if the corresponding index
    is the index of a root.

    Note: The size of an up-tree is the number of nodes in it
    '''

    def __init__(self, n):
        '''Initializes a forest of n up-trees with each node as a root'''
        self.parents = np.arange(n) # Returns a list of numbers 1 to n
        self.sizes = np.ones(n) # Returns a list with n number of ones


    def union(self, i, j):
        '''Unites the up-trees which contain indices i and j
        Does union by size: the smaller up-tree is linked to the
        root of the larger up-tree'''
        root_i, root_j = self.find(i), self.find(j)
        if root_i == root_j:
            return root_i
        large, small = root_i, root_j
        if self.sizes[root_i] < self.sizes[root_j]:
            large, small = small, large

        self.sizes[large] += self.sizes[small]
        self.sizes[small] = 0
        self.parents[small] = large
        return large


    def find(self, i):
        '''Returns the index of the root node of the subtree node i belongs to
        Does path-compression: links all ancestral nodes directly to the root'''
        if self.parents[i] == i:
            return i
        else:
            self.parents[i] = self.find(self.parents[i])
            return self.parents[i]


    def groups(self):
        '''Returns a list of disjoint lists where each disjoint list contains the indices of all nodes in the corresponding up-tree'''
        roots = map(self.find, self.parents)
        groups = defaultdict(list)
        for i, root in enumerate(roots):
            groups[root].append(i)
        return list(groups.values())
