# ~~~~~~
# fdt.py
# ~~~~~~

import copy
import numpy as np
from fuzzyset import FuzzySet
from fuzzyrule import FuzzyRule, FuzzyControlSystem
from fuzzyvariable import FuzzyVariable
from fuzzyplotter import FuzzyPlotter

import warnings
warnings.filterwarnings("ignore")

class Node(object):

    def __init__(self, depth, place, state, node_samples,
                 node_fuzzyset=None, node_parent=None, node_path=[]):
        self.depth = depth
        self.place = place
        self.state = state
        self.node_samples = node_samples
        self.node_fuzzyset = node_fuzzyset
        self.node_parent= node_parent
        self.node_path = node_path

        self.info = {'split_attribute': None,
                     'split_ambiguity': None,
                     'child_ambiguity': None,
                     'child_samples': None,
                     'child_fuzzy': None,
                     'truth_level': None,
                     'parent_places': None}

    def _get_info(self, info):
        self.info = info


class FuzzyTree(object):

    def __init__(self, fuzzy_dataset, fuzzy_variables, acut=0.5, tlvl=0.75, maxdepth=None, min_samples_split=2):
        # Initialize the fuzzy dataset and its dimensions
        self.data = copy.deepcopy(fuzzy_dataset)
        self.fvar = copy.deepcopy(fuzzy_variables)
        self.M = len(self.data[0])              # number of samples
        self.N = len(fuzzy_variables) - 1       # number of attributes
        self.C = len(fuzzy_variables[-1].fuzzy) # number of fuzzy classes
        # Initialize the hyper-parameters
        self.a = acut
        self.tlvl = tlvl
        if maxdepth is None:
            self.maxdepth = self.N - 1
        else:
            self.maxdepth = maxdepth
        self.min_samples_split= min_samples_split
        # Initialize object attributes
        self.the_tree = {}

    def build_tree(self):
        self._build_root()
        self._build_tree()
     
    def _build_root(self):
        """Build the root node."""
        root_samples = np.arange(0, self.M)
        root = Node(0, 0, 'root', root_samples)
        self._get_split_root(root)
        self.the_tree[0] = [root]
        
    def _get_split_root(self, node):
        """Split the root node."""
        cur_ambiguity, node_gval = self._get_ambiguity(node)
        best_attribute = cur_ambiguity.index(min(cur_ambiguity))
        best_ambiguity = cur_ambiguity[best_attribute]
##        print(cur_ambiguity,best_ambiguity)
##        input()
        info = {'split_attribute': best_attribute,
                'split_ambiguity': best_ambiguity,
                'child_ambiguity': node_gval[best_attribute],
                'child_samples': [],
                'child_fuzzy': [],
                'truth_level': [],
                'parent_places': []}
        
        node._get_info(info)
        node.info['child_samples'] = self._get_nodes(node)
        node.info['child_fuzzy'] = self._get_fuzzy(node, best_attribute)
        node.info['truth_level'] = self._get_truth_level(node)

    def _build_tree(self):
        depth = 1
        while depth <= self.maxdepth:
            # Build the next layer if the current layer contains at least one node.
            if self._check_layer(depth):
                self.maxdepth = depth
                break
            else:
                self._build_layer(depth)
            # Termination criterion based on max depth
            if depth == self.maxdepth:
                self._make_leaves()
                break
            else:
                depth += 1
        self._get_fuzzy_rules()

    def _check_layer(self, depth):
        for node in self.the_tree[depth-1]:
            if node.state == 'node' or node.state == 'root':
                return False
        else:
            return True

    def _make_leaves(self):
        for node in self.the_tree[self.maxdepth]:
            pnode = node.node_parent
            node.state = 'leaf'
            node.info['split_attribute'] = None
            node.info['split_ambiguity'] = pnode.info['child_ambiguity'][node.place]
            node.info['child_ambiguity'] = None
            node.info['child_samples'] = pnode.info['child_samples'][node.place]
            node.info['child_fuzzy'] = None
            node.info['truth_level'] =pnode.info['truth_level'][node.place]
            
    def _build_layer(self, depth):
        self.the_tree[depth] = []
        for parent in self.the_tree[depth-1]:

            cur_path = copy.deepcopy(parent.node_path)
            cur_path.append(parent.info['split_attribute'])

            
            # If the parent node is a leaf, go to the next node of the current layer
            if parent.state == 'leaf': continue

            # Create the child nodes
            for k in parent.info['child_samples'].keys():

                parent_places = copy.deepcopy(parent.info['parent_places'])
                parent_places.append(k)
                
                # Build the k-th node
                node = Node(depth, k, 'node', parent.info['child_samples'][k],
                            parent.info['child_fuzzy'][k], node_parent=parent, node_path=cur_path)
             
                # Check whether the current node is a leaf
                if node.node_samples.size <= self.min_samples_split:
                    node.state = 'leaf'
                    
                if (parent.info['truth_level'][k] >= self.tlvl*np.ones(len(parent.info['truth_level'][k]))).any():
                    node.state = 'leaf'

                if parent.info['child_ambiguity'][k] == 0:
                    node.state = 'leaf'

                # Get node info
                if node.state == 'leaf':
                    info = {'split_attribute': None,
                            'split_ambiguity': parent.info['child_ambiguity'][k],
                            'child_ambiguity': None,
                            'child_samples': parent.info['child_samples'][k],
                            'child_fuzzy': None,
                            'truth_level': parent.info['truth_level'][k],
                            'parent_places': parent_places}
                    node._get_info(info)
                    
                else:
                    cur_ambiguity, node_gval = self._get_ambiguity(node)
                    # TODO
                    if (min(cur_ambiguity) >= parent.info['child_ambiguity'][k]+0) or parent.info['child_ambiguity'][k] == 0:
                        node.state ='leaf'
                        info = {'split_attribute': None,
                                'split_ambiguity': parent.info['child_ambiguity'][k],
                                'child_ambiguity': None,
                                'child_samples': parent.info['child_samples'][k],
                                'child_fuzzy': None,
                                'truth_level': parent.info['truth_level'][k],
                                'parent_places': parent_places}
                        node._get_info(info)
                        
                    else:
                        idx_best_amb = cur_ambiguity.index(min(cur_ambiguity))
                        split_attr = [i for i in range(self.N) if i not in node.node_path]
                        best_attribute = split_attr[idx_best_amb]
                        best_ambiguity = min(cur_ambiguity)
                        info = {'split_attribute': best_attribute,
                                'split_ambiguity': best_ambiguity,
                                'child_ambiguity': node_gval[idx_best_amb],
                                'child_samples': None,
                                'child_fuzzy': None,
                                'truth_level': None,
                                'parent_places': parent_places}
                        node._get_info(info)
                        node.info['child_samples'] = self._get_nodes(node)
                        node.info['child_fuzzy'] = self._get_fuzzy(node, best_attribute)
                        node.info['truth_level'] = self._get_truth_level(node)
##                print(cur_ambiguity,best_ambiguity)
##                input()
                self.the_tree[depth].append(node)

    def _get_nodes(self, node):
        """Get the samples for every child node."""
        dt = copy.deepcopy(self.data[node.info['split_attribute']].T)
        dt[dt < self.a] = 0
        nodes = {}
        for i in range(len(self.fvar[node.info['split_attribute']])):
            a = set(node.node_samples)
            b = set(list(np.where(dt[i] > 0)[0]))
            nodes[i] = np.array(list(a.intersection(b)))
            if nodes[i].size == 0: del nodes[i]
        return nodes

    def _get_fuzzy(self, node, best_attr):
        if node.state == 'root':
            idx = node.info['child_samples'].keys()
            fz = {i: self.fvar[best_attr].fuzzy[i] for i in idx}
            return fz
        elif node.state == 'node':
            idx = node.info['child_samples'].keys()
            fnow = node.node_fuzzyset
            fnew = {i: fnow.mintersect(f) for i, f in zip(idx, self.fvar[best_attr].fuzzy)}
            return fnew

    def _get_truth_level(self, node):
        t = {}
        ind = node.node_samples
        fuzz = node.info['child_fuzzy']    
        for i, f in zip(fuzz.keys(), fuzz.values()):
            temp = np.array([f.dsub(c) for c in self.fvar[-1].fuzzy])
            temp = np.around(temp, decimals=2)
            t[i] = temp
        return t
    
    def _get_ambiguity(self, node):
        g, w = self._get_pdist(node)
        temp = g * w
        amb = []
        for row in temp:
            amb.append(np.sum(row)**1)
        #print(amb, '\n', g, '\n', w, '\n')
        return amb, g
    
    def _get_pdist(self, node):
        Gattr = []
        Wattr = []
        if node.state == 'root':
            for i, fv0 in enumerate(self.fvar.values()):
                if i == self.N: break
                fv = copy.deepcopy(fv0)
                fv.acut(self.a)
                Gval = np.zeros(len(fv.fuzzy))
                Wval = np.zeros(len(fv.fuzzy))
                for j, fuzz in enumerate(fv.fuzzy):
                    pdist = np.array([fuzz.dsub(tf) for tf in self.fvar[-1].fuzzy])
                    Gval[j] = self._calc_ambiguity(pdist)
                    Wval[j] = fuzz.card()
                Wval /= np.sum(Wval)
                Wattr.append(Wval)
                Gattr.append(Gval)
        else:
            p_node = node.node_parent
            ind = p_node.info['child_samples'][node.place]
            nowvar = copy.deepcopy(self.fvar[p_node.info['split_attribute']])
            #nowvar.acut(self.a)
            f = nowvar.fuzzy[node.place]
            to_search = [i for i in range(self.N) if i not in node.node_path ]
            for j in to_search:
                fv = copy.deepcopy(self.fvar[j])
                #fv.acut(self.a)
                Gval = np.zeros(len(fv.fuzzy))
                Wval = np.zeros(len(fv.fuzzy))
                for j, fuzz in enumerate(fv.fuzzy):
                    nowf = f.mintersect(fuzz)
                    pdist = np.array([nowf.dsub_ind(tf, ind) for tf in self.fvar[-1].fuzzy])
                    if np.max(pdist) == 0:
                        Gval[j] = 0
                    else:
                        Gval[j] = self._calc_ambiguity(pdist)
                    Wval[j] = f.mintersect_ind(fuzz, ind).card()
                try:
                    Wval /= np.sum(Wval)
                except:
                    Wval = 0
                Wattr.append(Wval)
                Gattr.append(Gval)            
        return np.array(Gattr), np.array(Wattr)
    
    def _calc_ambiguity(self, row):
        try:
            row = np.append(row, 0) / np.max(row)
        except:
            pass
        else:
            row = -np.sort(-row, axis=0)
            diff = -np.diff(row)
            idx = np.arange(1, row.shape[0])
            pval = np.sum(diff * np.log(idx))
        return pval

    def _get_fuzzy_rules(self):
        self.rules = []
        self.antecedent = []
        self.consequent = []
        self.sdeg = []
        self.rules_text_format = []
        self.rules_fis_format = []
        for layer in self.the_tree.values():
            for node in layer:
                if node.state == 'leaf':
                    # TODO
                    if max(node.info['truth_level']) > 0:
                        ant, con, sdeg = self._get_rule(node)
                        self.antecedent.append(ant)
                        self.consequent.append(con)
                        self.sdeg.append(sdeg)
                        self.rules_text_format.append(self._get_rule_text(node))
                        self.rules_fis_format.append(self._get_rule(node))
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

    def _get_rule_fis(self, node):
        ante = ''
        for p, f in zip(node.node_path, node.info['parent_places']):
            ante += '{}["{}"]'.format(self.fvar[p].name, self.fvar[p].terms[f])
            ante += ' & '
        ante = ante[:-3]
        idx = np.argmax(node.info['truth_level'])
        cons = '{}["{}"]'.format(self.fvar[-1].name, self.fvar[-1].terms[idx])
        rule = FuzzyRule(ante, cons)
        return rule

    def print_rules(self):
        for r in self.rules_text_format:
            print(r)

    def predict(self, test):
        res = []
        for row in test:
            res.append(self._predict_row(row))
        return res

    def _predict_row(self, row):
        rule_membership = []
        for i, rule in enumerate(self.antecedent):
            mdeg = []
            for idx, f in rule:
                mdeg.append(row[idx][f])
##                try:
##                    mdeg.append(row[idx][f])
##                except IndexError:
##                    mdeg.append(0)
##                    print(row, len(row), idx, f, rule, mdeg)
##                    input()
            rule_membership.append(min(mdeg))
        res = {i: [] for i in range(self.C)}
        for j, m in zip(self.consequent, rule_membership):
            res[j].append(m)
        for j in res.keys():
            try:
                res[j] = max(res[j])
            except ValueError:
                res[j] = 0
        return res
    
    def to_single_class(self, res):
        cls = []
        for d in res:
            temp = list(d.values())
            idx = temp.index(max(temp))
            cls.append(idx)
        return cls

    def accuracy(self, pred, act):
        pred = np.array(pred)
        act = np.array(act)
        idx_correct = np.argwhere(pred == act)
        idx_wrong = np.argwhere(pred != act)
        acc = idx_correct.size / act.size
        return acc, idx_correct, idx_wrong
        
    def print_tree(self):
        for layer in self.the_tree.values():
            for obj in layer:
                print('Node: {}'.format(obj.state))
                if obj.state != 'root':
                    print('depth: {}, place: {}, parent_id: {}'.format(obj.depth, obj.place, obj.node_parent.place))
                else:
                    print('depth: {}, place: {}, parent_id: {}'.format(obj.depth, obj.place, None))
                print(obj.info)
                print()

    def regress(self, test, yt):
        predicted = []
        for k, row in enumerate(test):
            act_deg = []
            fparts = []
            for i, rule in enumerate(self.rules):
                ante, cons = rule[0], rule[-1]
                temp = []
                for tup in ante:
                    t = row[tup[0]][tup[-1]]
                    temp.append(t)
                act_deg.append(min(temp))
                if act_deg[i] == 0:
                    if i == 0:
                        f = self.fvar[-1].fuzzy[cons]
                        s = 'f.func.{}(f.num_x, f.prmts)'.format(f.mfunc)
                        newm = eval(s)
                        newf = FuzzySet(f.num_x, newm)
                        newf.cutpoint(act_deg[i])
                        self.fout = newf
                    continue
                else:
                    f = self.fvar[-1].fuzzy[cons]
                    s = 'f.func.{}(f.num_x, f.prmts)'.format(f.mfunc)
                    newm = eval(s)
                    newf = FuzzySet(f.num_x, newm)
                    newf.cutpoint(act_deg[i])
                    if i == 0:
                        self.fout = newf
                    else:
                        self.fout = self.fout | newf
            # mean of max method
##            dummy = self.fout.x[np.where(self.fout.m >= max(self.fout.m))[0]]
##            val = sum(dummy)/len(dummy)
            # center of gravity method
            val = self.centroid(self.fout.x, self.fout.m)
            predicted.append(val)
##            print('## Actual value: {}'.format(yt[k]))
##            print('## Regressed value: {}'.format(val))
##            FuzzyPlotter(self.fout) 
        diff = [(a-p)**2 for a, p in zip(yt, predicted)]
        rmse = sum(diff) / len(diff)
        return predicted, rmse


    def centroid(self, x, mfx):
        """
        Defuzzification using centroid (`center of gravity`) method.

        Parameters
        ----------
        x : 1d array, length M
            Independent variable
        mfx : 1d array, length M
            Fuzzy membership function

        Returns
        -------
        u : 1d array, length M
            Defuzzified result

        See also
        --------
        skfuzzy.defuzzify.defuzz, skfuzzy.defuzzify.dcentroid
        """

        '''
        As we suppose linearity between each pair of points of x, we can calculate
        the exact area of the figure (a triangle or a rectangle).
        '''

        sum_moment_area = 0.0
        sum_area = 0.0

        # If the membership function is a singleton fuzzy set:
        if len(x) == 1:
            return x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float)

        # else return the sum of moment*area/sum of area
        for i in range(1, len(x)):
            x1 = x[i - 1]
            x2 = x[i]
            y1 = mfx[i - 1]
            y2 = mfx[i]

            # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
            if not(y1 == y2 == 0.0 or x1 == x2):
                if y1 == y2:  # rectangle
                    moment = 0.5 * (x1 + x2)
                    area = (x2 - x1) * y1
                elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                    moment = 2.0 / 3.0 * (x2-x1) + x1
                    area = 0.5 * (x2 - x1) * y2
                elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                    moment = 1.0 / 3.0 * (x2 - x1) + x1
                    area = 0.5 * (x2 - x1) * y1
                else:
                    moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                    area = 0.5 * (x2 - x1) * (y1 + y2)

                sum_moment_area += moment * area
                sum_area += area

        return sum_moment_area / np.fmax(sum_area,
                                         np.finfo(float).eps).astype(float)
        
    def bisector(self, x, mfx):
        """
        Defuzzification using bisector, or division of the area in two equal parts.

        Parameters
        ----------
        x : 1d array, length M
            Independent variable
        mfx : 1d array, length M
            Fuzzy membership function

        Returns
        -------
        u : 1d array, length M
            Defuzzified result

        See also
        --------
        skfuzzy.defuzzify.defuzz
        """
        '''
        As we suppose linearity between each pair of points of x, we can calculate
        the exact area of the figure (a triangle or a rectangle).
        '''
        sum_area = 0.0
        accum_area = [0.0] * (len(x) - 1)

        # If the membership function is a singleton fuzzy set:
        if len(x) == 1:
            return x[0]

        # else return the sum of moment*area/sum of area
        for i in range(1, len(x)):
            x1 = x[i - 1]
            x2 = x[i]
            y1 = mfx[i - 1]
            y2 = mfx[i]

            # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
            if not(y1 == y2 == 0. or x1 == x2):
                if y1 == y2:  # rectangle
                    area = (x2 - x1) * y1
                elif y1 == 0. and y2 != 0.:  # triangle, height y2
                    area = 0.5 * (x2 - x1) * y2
                elif y2 == 0. and y1 != 0.:  # triangle, height y1
                    area = 0.5 * (x2 - x1) * y1
                else:
                    area = 0.5 * (x2 - x1) * (y1 + y2)
                sum_area += area
                accum_area[i - 1] = sum_area

        # index to the figure which cointains the x point that divide the area of
        # the whole fuzzy set in two
        index = np.nonzero(np.array(accum_area) >= sum_area / 2.)[0][0]

        # subarea will be the area in the left part of the bisection for this set
        if index == 0:
            subarea = 0
        else:
            subarea = accum_area[index - 1]
        x1 = x[index]
        x2 = x[index + 1]
        y1 = mfx[index]
        y2 = mfx[index + 1]

        # We are interested only in the subarea inside the figure in which the
        # bisection is present.
        subarea = sum_area/2. - subarea

        x2minusx1 = x2 - x1
        if y1 == y2:  # rectangle
            u = subarea/y1 + x1
        elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
            root = np.sqrt(2. * subarea * x2minusx1 / y2)
            u = (x1 + root)
        elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
            root = np.sqrt(x2minusx1*x2minusx1 - (2.*subarea*x2minusx1/y1))
            u = (x2 - root)
        else:
            m = (y2-y1) / x2minusx1
            root = np.sqrt(y1*y1 + 2.0*m*subarea)
            u = (x1 - (y1-root) / m)
        return u




### Testing/Debugging
##from fuzzyplotter import FuzzyPlotter
##
### Universe of discourse
##x = np.arange(1,17)
##
### Outlook variable  
##outlook = FuzzyVariable('Outlook', x, terms=['Sunny', 'Cloudy', 'Rain'])
##m_outlook = np.array([[0.9, 0.8, 0, 0.2, 0, 0, 0, 0, 1, 0.9, 0.7, 0.2, 0.9, 0, 0, 1],
##             [0.1, 0.2, 0.7, 0.7, 0.1, 0.7, 0.3, 1, 0, 0.1, 0.3, 0.6, 0.1, 0.9, 0, 0],
##             [0, 0, 0.3, 0.1, 0.9, 0.3, 0.7, 0, 0, 0, 0, 0.2, 0, 0.1, 1, 0]])
##outlook.setmval(m_outlook.T)
##
### Temperature variable
##temperature = FuzzyVariable('Temperature', x, terms=['Hot', 'Mild', 'Cool'])
##m_temp = np.array([[1, 0.6, 0.8, 0.3, 0.7, 0, 0, 0, 1, 0, 1, 0, 0.2, 0, 0, 0.5],
##          [0, 0.4, 0.2, 0.7, 0.3, 0.3, 0, 0.2, 0, 0.3, 0, 1, 0.8, 0.9, 0, 0.5],
##          [0, 0, 0, 0, 0, 0.7, 1, 0.8, 0, 0.7, 0, 0, 0, 0.1, 1, 0]])
##temperature.setmval(m_temp.T)
##
### Humidity variable
##humidity = FuzzyVariable('Humidity', x, terms=['Humid', 'Normal'])
##m_hum= np.array([[0.8, 0, 0.1, 0.2, 0.5, 0.7, 0, 0.2, 0.6, 0, 1, 0.3, 0.1, 0.1, 1, 0],
##        [0.2, 1, 0.9, 0.8, 0.5, 0.3, 1, 0.8, 0.4, 1, 0, 0.7, 0.9, 0.9, 0 ,1]])
##humidity.setmval(m_hum.T)
##
### Wind variable
##wind = FuzzyVariable('Wind', x, terms=['Windy', 'Not_windy'])
##m_wind= np.array([[0.4, 0, 0.2, 0.3, 0.5, 0.4, 0.1, 0, 0.7, 0.9, 0.2, 0.3, 1, 0.7, 0.8, 0],
##         [0.6, 1, 0.8, 0.7, 0.5, 0.6, 0.9, 1, 0.3, 0.1, 0.8, 0.7, 0, 0.3, 0.2, 1]])
##wind.setmval(m_wind.T)
##
### Plan variable
##plan = FuzzyVariable('Plan', x, terms=['Volleyball', 'Swimming', 'Weight_lifting'])
##m_plan= np.array([[0, 1, 0.3, 0.9, 0, 0.2, 0, 0.7, 0.2, 0, 0.4, 0.7, 0, 0, 0, 0.8],
##         [0.8, 0.7, 0.6, 0.1, 0, 0, 0, 0, 0.8, 0.3, 0.7, 0.2, 0, 0, 0, 0.6],
##         [0.2, 0, 0.1, 0, 1, 0.8, 1, 0.3, 0, 0.7, 0, 0.1, 1, 1, 1, 0]])
##plan.setmval(m_plan.T)
##
### Build the fuzzy dataset
##db = {0: copy.deepcopy(m_outlook.T),
##      1: copy.deepcopy(m_temp.T),
##      2: copy.deepcopy(m_hum.T),
##      3: copy.deepcopy(m_wind.T),
##      -1: copy.deepcopy(m_plan.T)}
##
##db_test = [[db[i][j] for i in range(4)] for j in range(16)]
##
##fvar = {0: outlook,
##        1: temperature,
##        2: humidity,
##        3: wind,
##        -1: plan}
##
### Build a FuzzyTree
##alpha = 0.5
##fdt = FuzzyTree(db, fvar, acut=alpha, tlvl=0.7)
##fdt.build_tree()
##fdt.print_tree()
##fdt.print_rules()
##r = fdt.predict(db_test)
##c = fdt.to_single_class(r)
##
##fdt.regress(db_test)
