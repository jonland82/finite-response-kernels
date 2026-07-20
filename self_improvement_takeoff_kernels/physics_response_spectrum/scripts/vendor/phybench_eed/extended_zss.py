#!/usr/bin/env python
"""Extended Zhang-Shasha tree distance from the official PHYBench evaluator.

Vendored from phybench-official/phybench commit
d9db3ec7246f3678aaee65d44a32649ad93beea2. The implementation is kept
local so the experiment can be reproduced without a mutable dependency.
"""

import collections

from numpy import ones, zeros


class Node(object):
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or list()

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def addkid(self, node, before=False):
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def get(self, label):
        if self.label == label:
            return self
        for child in self.children:
            if label in child:
                return child.get(label)


class AnnotatedTree(object):
    def __init__(self, root, get_children):
        self.get_children = get_children
        self.root = root
        self.nodes = []
        self.ids = []
        self.lmds = []
        self.keyroots = None

        stack = [(root, collections.deque())]
        pstack = []
        node_id = 0
        while stack:
            node, ancestors = stack.pop()
            current_id = node_id
            for child in self.get_children(node):
                child_ancestors = collections.deque(ancestors)
                child_ancestors.appendleft(current_id)
                stack.append((child, child_ancestors))
            pstack.append(((node, current_id), ancestors))
            node_id += 1

        leftmost = {}
        keyroots = {}
        index = 0
        while pstack:
            (node, current_id), ancestors = pstack.pop()
            self.nodes.append(node)
            self.ids.append(current_id)
            if not self.get_children(node):
                lmd = index
                for ancestor in ancestors:
                    if ancestor not in leftmost:
                        leftmost[ancestor] = index
                    else:
                        break
            else:
                lmd = leftmost[current_id]
            self.lmds.append(lmd)
            keyroots[lmd] = index
            index += 1
        self.keyroots = sorted(keyroots.values())


def ext_distance(
    tree_a,
    tree_b,
    get_children,
    single_insert_cost,
    insert_cost,
    single_remove_cost,
    remove_cost,
    update_cost,
):
    """Compute extended tree-edit distance between two expression trees."""
    tree_a = AnnotatedTree(tree_a, get_children)
    tree_b = AnnotatedTree(tree_b, get_children)
    size_a = len(tree_a.nodes)
    size_b = len(tree_b.nodes)
    tree_distances = zeros((size_a, size_b), float)
    forest_distances = 1000 * ones((size_a + 1, size_b + 1), float)

    def tree_distance(x, y):
        a_lmds = tree_a.lmds
        b_lmds = tree_b.lmds
        a_nodes = tree_a.nodes
        b_nodes = tree_b.nodes

        forest_distances[a_lmds[x]][b_lmds[y]] = 0
        for i in range(a_lmds[x], x + 1):
            node = a_nodes[i]
            forest_distances[i + 1][b_lmds[y]] = (
                forest_distances[a_lmds[i]][b_lmds[y]] + remove_cost(node)
            )

        for j in range(b_lmds[y], y + 1):
            node = b_nodes[j]
            forest_distances[a_lmds[x]][j + 1] = (
                forest_distances[a_lmds[x]][b_lmds[j]] + insert_cost(node)
            )

        for i in range(a_lmds[x], x + 1):
            for j in range(b_lmds[y], y + 1):
                node_a = a_nodes[i]
                node_b = b_nodes[j]
                costs = [
                    forest_distances[i][j + 1] + single_remove_cost(node_a),
                    forest_distances[i + 1][j] + single_insert_cost(node_b),
                    forest_distances[a_lmds[i]][j + 1] + remove_cost(node_a),
                    forest_distances[i + 1][b_lmds[j]] + insert_cost(node_b),
                ]
                cheapest = min(costs)
                if a_lmds[x] == a_lmds[i] and b_lmds[y] == b_lmds[j]:
                    tree_distances[i][j] = min(
                        cheapest,
                        forest_distances[i][j] + update_cost(node_a, node_b),
                    )
                    forest_distances[i + 1][j + 1] = tree_distances[i][j]
                else:
                    forest_distances[i + 1][j + 1] = min(
                        cheapest,
                        forest_distances[a_lmds[i]][b_lmds[j]]
                        + tree_distances[i][j],
                    )

    for x in tree_a.keyroots:
        for y in tree_b.keyroots:
            tree_distance(x, y)
    return tree_distances[-1][-1]
