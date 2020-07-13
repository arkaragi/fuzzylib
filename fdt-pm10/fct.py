# ~~~~~~
# fdt.py
# ~~~~~~

import copy
import numpy as np
from fuzzyset import FuzzySet
from fuzzyvariable import FuzzyVariable


class Node(object):

    def __init__(self, depth, place, state, node_ind, node_feat, node_path, fuzz, info={}):
        self.depth = depth
        self.place = place
        self.state = state
        self.node_ind = node_ind
        self.node_feat = node_feat
        self.node_path = node_path
        self.fuzz = fuzz
        self.info = info

    def get_info(self, info):
        self.info = info
        self.split_feature = info['split_feature']
        self.entropy = info['entropy']
        self.parent_id = info['parent_id']
        self.split_ind = info['split_ind']
        self.path_features = info['path_features']
        self.fuzzy_sets = info['fuzzy_sets']
        self.truth_level = info['truth_level']


class FuzzyTreeGID3(object):

    """Base class for a Fuzzy Decision Tree.

        This implementation is based on the Fuzzy ID3 algorithm.

        Parameters
        ----------
        fuzzy_dataset : dict
            The fuzzified version of the initial dataset. Every key
            refers to a fuzzy variable in the order given in fuzzy_variables
            list. Every key contains Cj arrays of memebership grades, where Cj
            is the number of fuzzy sets of the jth variable.

        fuzzy_variables : list
            Contains the FuzzyVariable objects.

        acut : float, [0, 1]
            Defines the a-cut.
    """

    def __init__(self, fuzzy_dataset, fuzzy_variables, acut=0, tlvl=0.85):
        self.data = fuzzy_dataset
        self.fvar = fuzzy_variables
        self.a = acut
        self.tmax = tlvl
        self.M = len(self.data[0])
        self.N = len(fuzzy_variables) - 1
        self.C = len(fuzzy_variables[-1].fuzzy)
        self.the_tree = {}
        self.build_tree()

    def build_tree(self):
        self.build_root()
        self.max_depth = 1
        while self.max_depth < 3:
            self.build_next_layer()
            self.max_depth += 1

    def build_root(self):
        root_ind = np.arange(0, self.M)
        root_feat = list(range(self.N))
        root = Node(0, 0, 'root', root_ind, root_feat, [], -1)
        info = self.get_split(root)
        root.get_info(info)
        self.the_tree[root.depth] = [root]

    def build_next_layer(self):
        self.the_tree[self.max_depth] = []
        prev_nodes = self.the_tree[self.max_depth-1]
        for pnode in prev_nodes:
            if pnode.state == 'terminal': continue
            ind = pnode.info['split_ind']
            for k in range(len(ind)):
                if ind[k].size == 0: continue
                # get available features for the current node
                cur_node_feat = copy.deepcopy(pnode.node_feat)
                cur_node_feat.remove(pnode.info['split_feature'])
                # get the path of the current node
                pth = copy.deepcopy(pnode.info['path_features'])
                pth.append(pnode.info['split_feature'])
                # get the fuzzy sets of the current node
                z = pnode.info['split_feature']
                w = self.fvar[z].terms[k]
                fuzz = copy.deepcopy(pnode.info['fuzzy_sets'])
                fuzz.append(self.fvar[z][w])
                # build the Node
                node = Node(self.max_depth, k, 'node', ind[k], cur_node_feat, pth, fuzz)
                if node.node_feat == []:
                    node.state = 'terminal'
                info = self.get_split(node)
##                info['parent_id'] = str(pnode.info['parent_id']) + str(pnode.place)
                info['parent_id'] = str(pnode.info['parent_id']) + str(pnode.place)
                pfeat = copy.deepcopy(pnode.info['path_features'])
                pfeat.append(pnode.info['split_feature'])
                info['path_features'] = pfeat
                z = pnode.info['split_feature']
                w = self.fvar[z].terms[node.place]
                info['fuzzy_sets'] = copy.deepcopy(pnode.info['fuzzy_sets'])
                info['fuzzy_sets'].append(self.fvar[z][w])
                node.get_info(info)
                self.the_tree[self.max_depth].append(node)
##        self._get_fuzzy_rules()

                print(node.state, node.place, node.info['parent_id'])
                print(node.info['truth_level'], node.info, sep='\n')
                input()
                
    def get_split(self, node):
        t = 0
        if node.state != 'root':
            t = self.get_truth_factor(node)
            if (t >= self.tmax).any() or (node.state=='terminal'):
                node.state = 'terminal'
                info = {'split_feature': -1,
                        'entropy': 0,
                        'split_ind': -1,
                        'parent_id': 0,
                        'path_features': [],
                        'fuzzy_sets': [],
                        'truth_level': t}
                return info
        ind = node.node_ind
        flist = node.node_feat
        # get the feature with the lowest split entropy
        eV = []
        for i in flist:
            e = self.get_fuzzy_entropy(i, ind)
            eV.append(e)
        best = eV.index(min(eV))
        best_j = flist[best]
        # get the samples of each child node
        dt = copy.deepcopy(self.data[best_j].T)
        #dt[dt <= self.a] = 0
        nodes = {}
        for i in range(len(self.fvar[best_j].fuzzy)):
            a = set(ind)
            b = set(list(np.where(dt[i] > 0)[0]))
            nodes[i] = np.array(list(a.intersection(b)))
            nodes[i].sort()    
        info = {'split_feature': best_j,
                'entropy': eV[best],
                'split_ind': nodes,
                'parent_id': 0,
                'path_features': [],
                'fuzzy_sets': [],
                'truth_level': t}
        return info

    def get_truth_factor(self, node):
        # get the truth factor for every new fuzzy set
        ind = node.node_ind
        path = node.node_path
        tV = []
        to_split = []
        for p, f in zip(path, node.fuzz):
            temp = f.m[ind]
            to_split.append(temp)
        to_split = np.array(to_split)
        cls_val = self.data[-1][ind]
        for c in range(self.C):
            a = np.sum(np.minimum(np.min(to_split, axis=0), cls_val[:, c].T))
            b = np.sum(np.min(to_split, axis=0))
            tV.append(a / b)
        return np.array(tV)
    
    def get_beta_coef(self, j, ind=()):
        to_split = self.data[j][ind]
        cls_val = self.data[-1][ind]
        # get beta coefficients
        bV = []
        for c in range(self.C):
            val = np.minimum(to_split.T, cls_val[:, c]).T
            b = np.sum(val, axis=0) / (np.sum(to_split, axis=0) + 1e-9) + 1e-6
            bV.append(b)
        # get w-coefficients
        wV = np.sum(to_split.T, axis=1) / np.sum(to_split)
        return np.array(bV).T, np.array(wV).T
    
    def get_fuzzy_entropy(self, j, ind=()):
        # get the fuzzy entropy for the j-th feature
        bV, wV = self.get_beta_coef(j, ind)
        eV = -np.sum(bV*np.log2(bV), axis=1)
        total_entropy = np.sum(wV.dot(eV))
        return total_entropy
    
    def _get_fuzzy_rules(self):
        self.rules = []
        self.antecedent = []
        self.consequent = []
        self.sdeg = []
        self.rules_text_format = []
        for layer in self.the_tree.values():
            for node in layer:
                if node.state == 'terminal':
                    print(node.info)
                    input()
                    # TODO
                    if max(node.info['truth_level']) > 0:
                        ant, con, sdeg = self._get_rule(node)
                        self.antecedent.append(ant)
                        self.consequent.append(con)
                        self.sdeg.append(sdeg)
                        self.rules_text_format.append(self._get_rule_text(node))
                        self.rules.append([ant, con])

    def _get_rule(self, node):
        antecedent = []
        for p, f in zip(node.node_path, node.info['parent_places']):
            antecedent.append((p, f))
        consequent = np.argmax(node.info['truth_level'])
        return antecedent, consequent, np.max(node.info['truth_level'])
            
    def _get_rule_text(self, node):
        rule = 'If '
        for p, f in zip(node.node_path, node.info['parent_places']):
            rule += '{} is "{}" '.format(self.fvar[p].name, self.fvar[p].terms[f])
            rule += 'AND '
        rule = rule[:-4]
        idx = np.argmax(node.info['truth_level'])
        rule += 'THEN {} (S={})'.format(self.fvar[-1].terms[idx], node.info['truth_level'][idx])
        return rule

    def print_rules(self):
        for r in self.rules_text_format:
            print(r)

                
### Testing/Debugging
##from fuzzifier import Fuzzifier
##from fuzzyplotter import FuzzyPlotter
##from sklearn.datasets import load_iris
### Initialize the dataset
##iris = load_iris()
##X = iris.data
##y = iris.target
##var_names = iris.feature_names
### Fuzzify the dataset
##fuzz = Fuzzifier(X, y)
##fuzz.get_clusters(cntrs=[3,4])
##fuzz.get_fuzzy_variables()
##fuzz.get_fuzzy_data()
##fdata = fuzz.fuzzy_dataset
### Plot the FuzzyDataset object
####FuzzyPlotter(fdata)
##
### Build a FuzzyTree
##fdt = FuzzyTree(X, fdata.fuzzy_dataset, fdata.fuzzy_variables)
##t = fdt.the_tree
##t0 = t[0][0]
##t1 = t[1][0]
##print(t1.info)
