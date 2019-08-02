#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# TODO: Use true labels and the preds to give statistics of where the method fails.

# TODO: Allow the user to select the parameters used for generating the validation data!
# TODO: After optimizing parameters, the execution on whole train graph is not efficient, several embds with same params

# TODO: EvalSetup checks (baselines exist, neigh param is correct, ee method exist, scores and report are correct)

from __future__ import division

import itertools
import os
import re
import subprocess
import warnings
import time

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

from evalne.evaluation import edge_embeddings
from evalne.evaluation import score
from evalne.evaluation import split
from evalne.methods import similarity as sim



class EvalSetup(object):
    r"""
    This class is a wrapper that parses the config file and provides the options as properties of the class.
    Also performs simple input checks

    Parameters
    ----------
    configpath : basestring
        The path of the configuration file.
    """

    def __init__(self, configpath):
        # Import config parser
        try:
            from ConfigParser import ConfigParser
        except ImportError:
            from configparser import ConfigParser

        # Read the configuration file
        config = ConfigParser()
        config.read(configpath)
        self._config = config

        # Check input paths
        self._check_inpaths()
        # Check methods
        self._check_methods('opne')
        self._check_methods('other')
        self._checkparams()
        self._check_edges()

    def _check_edges(self):
        if self.__getattribute__('train_frac') == 0:
            raise ValueError('The number of train edges can not be 0!')

        if (self.__getattribute__('num_fe_train') is not None and self.__getattribute__('num_fe_train') <= 0) or \
                (self.__getattribute__('num_fe_test') is not None and self.__getattribute__('num_fe_test') <= 0):
            raise ValueError('The number of train and test false edges has to be positive and grater than 0!')

    def _check_inpaths(self):
        l = len(self.__getattribute__('names'))
        for k in self._config.options('NETWORKS'):
            if len(self.__getattribute__(k)) != l:
                raise ValueError('Parameter `{}` in `NETWORKS` section does not have the required num. parameters ({})'
                                 .format(k, self.__getattribute__(k)))
        # Check if the input file exist
        for path in self.__getattribute__('inpaths'):
            if not os.path.exists(path):
                raise ValueError('Input network path {} does not exist'.format(path))

    def _check_methods(self, library):
        names = self.__getattribute__('names_' + library)
        methods = self.__getattribute__('methods_' + library)
        if names is not None and methods is not None and len(names) != len(methods):
            raise ValueError('Mismatch in the number of `NAMES` and `METHODS` to run in section `{} METHODS`'
                             .format(library))

    def _checkparams(self):
        directed = self.__getattribute__('directed')
        if self.__getattribute__('scores') == self.__getattribute__('maximize'):
            if sum(directed) == len(directed) or sum(directed) == 0:
                pass
            else:
                raise ValueError('Mix of directed and undirected networks found! Tabular output not supported.'
                                 'Please change the SCORES parameter to `all` or provide all dir./undir. networks')

    def getlist(self, section, option, dtype):
        r"""
        Returns option as a list of specified type, split by any kind of white space.

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.
        dtype : primitive type
            The type to which the output should be cast.

        Returns
        -------
        list : list
            A list of elements cast to the specified primitive type.
        """
        res = self._config.get(section, option).split()
        if len(res) == 0 or res[0] == '' or res[0] == 'None':
            return None
        else:
            return list(map(dtype, res))

    def getboollist(self, section, option):
        r"""
        Returns option as a list of booleans split by any kind of white space.
        Elements such as 'True', 'true', '1', 'yes', 'on' are considered True.
        Elements such as 'False', 'false', '0', 'no', 'off' are considered False.

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.

        Returns
        -------
        list : list
            A list of booleans.
        """
        res = self._config.get(section, option).split()
        if len(res) == 0 or res[0] == '' or res[0] == 'None':
            return None
        else:
            r = list()
            for elem in res:
                if elem in ['True', 'true', '1', 'yes', 'on']:
                    r.append(True)
                elif elem in ['False', 'false', '0', 'no', 'off']:
                    r.append(False)
            return r

    def getlinelist(self, section, option):
        r"""
        Returns option as a list of string, split specifically by a newline.

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.

        Returns
        -------
        list : list
            A list of strings.
        """
        res = self._config.get(section, option).split('\n')
        if len(res) == 0 or res[0] == '' or res[0] == 'None':
            return None
        else:
            return list(res)

    def getseplist(self, section, option):
        r"""
        Processes an options which contains a list of separators.
        Transforms \s, \t and \n to white space, tab and new line respectively

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.

        Returns
        -------
        list : list
            A list of strings.
        """
        separators = self.getlist(section, option, str)
        res = list()
        for sep in separators:
            s = sep.strip('\'')
            if s == '\\t':
                s = '\t'
            elif s == '\\s':
                s = ' '
            elif s == '\\n':
                s = '\n'
            res.append(s)
        return list(res)

    def gettuneparams(self, library):
        r"""
        Processes the tune parameters option. Generates a list of Nones the size of the number of methods.
        The list is filled in order with each line found in the TUNE_PARAMS option.

        Parameters
        ----------
        library : basestring
            This parameter indicates if the TUNE_PARAMETERS option processed if from OPNE METHODS of OTHER METHODS.

        Returns
        -------
        tune_params : list
            A list of string containing the parameters that need to be tuned.
        """
        methods = self.__getattribute__('methods_' + library)
        if library == 'opne':
            tune_params = self.getlinelist('OPENNE METHODS', 'tune_params_opne')
        elif library == 'other':
            tune_params = self.getlinelist('OTHER METHODS', 'tune_params_other')
        else:
            raise ValueError('Attribute name {}, does not exist'.format(library))
        if tune_params is None:
            tune_params = list()
        for i in range(len(methods) - len(tune_params)):
            tune_params.append(None)
        return tune_params

    @property
    def edge_embedding_methods(self):
        return self.getlist('GENERAL', 'edge_embedding_methods', str)

    @property
    def lp_model(self):
        return LogisticRegression(solver='liblinear')
        # return LogisticRegressionCV(Cs=10, cv=10, penalty='l2', scoring='roc_auc', solver='lbfgs', max_iter=200)

    @property
    def exp_repeats(self):
        return self._config.getint('GENERAL', 'exp_repeats')

    @property
    def embed_dim(self):
        return self._config.getint('GENERAL', 'embed_dim')

    @property
    def verbose(self):
        return self._config.getboolean('GENERAL', 'verbose')

    @property
    def seed(self):
        val = self._config.get('GENERAL', 'seed')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def names(self):
        return self.getlist('NETWORKS', 'names', str)

    @property
    def inpaths(self):
        return self.getlinelist('NETWORKS', 'inpaths')

    @property
    def outpaths(self):
        return self.getlinelist('NETWORKS', 'outpaths')

    @property
    def directed(self):
        return self.getboollist('NETWORKS', 'directed')

    @property
    def separators(self):
        return self.getseplist('NETWORKS', 'separators')

    @property
    def comments(self):
        return self.getseplist('NETWORKS', 'comments')

    @property
    def relabel(self):
        return self._config.getboolean('PREPROCESSING', 'relabel')

    @property
    def del_selfloops(self):
        return self._config.getboolean('PREPROCESSING', 'del_selfloops')

    @property
    def prep_nw_name(self):
        return self._config.get('PREPROCESSING', 'prep_nw_name')

    @property
    def write_stats(self):
        return self._config.getboolean('PREPROCESSING', 'write_stats')

    @property
    def delimiter(self):
        return self._config.get('PREPROCESSING', 'delimiter').strip('\'')

    @property
    def train_frac(self):
        return self._config.getfloat('TRAINTEST', 'train_frac')

    @property
    def fast_split(self):
        return self._config.getboolean('TRAINTEST', 'fast_split')

    @property
    def owa(self):
        return self._config.getboolean('TRAINTEST', 'owa')

    @property
    def num_fe_train(self):
        val = self._config.get('TRAINTEST', 'num_fe_train')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def num_fe_test(self):
        val = self._config.get('TRAINTEST', 'num_fe_test')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def traintest_path(self):
        return self._config.get('TRAINTEST', 'traintest_path')

    @property
    def lp_baselines(self):
        return self.getlinelist('BASELINES', 'lp_baselines')

    @property
    def neighbourhood(self):
        return self.getlist('BASELINES', 'neighbourhood', str)

    @property
    def names_opne(self):
        return self.getlist('OPENNE METHODS', 'names_opne', str)

    @property
    def methods_opne(self):
        return self.getlinelist('OPENNE METHODS', 'methods_opne')

    @property
    def tune_params_opne(self):
        return self.gettuneparams('opne')

    @property
    def names_other(self):
        return self.getlist('OTHER METHODS', 'names_other', str)

    @property
    def embtype_other(self):
        return self.getlist('OTHER METHODS', 'embtype_other', str)

    @property
    def write_weights_other(self):
        return self.getboollist('OTHER METHODS', 'write_weights_other')

    @property
    def write_dir_other(self):
        return self.getboollist('OTHER METHODS', 'write_dir_other')

    @property
    def methods_other(self):
        return self.getlinelist('OTHER METHODS', 'methods_other')

    @property
    def tune_params_other(self):
        return self.gettuneparams('other')

    @property
    def output_format_other(self):
        return self.getlinelist('OTHER METHODS', 'output_format_other')

    @property
    def input_delim_other(self):
        return self.getseplist('OTHER METHODS', 'input_delim_other')

    @property
    def output_delim_other(self):
        return self.getseplist('OTHER METHODS', 'output_delim_other')

    @property
    def maximize(self):
        return self._config.get('REPORT', 'maximize')

    @property
    def scores(self):
        return self._config.get('REPORT', 'scores')

    @property
    def curves(self):
        return self._config.get('REPORT', 'curves')

    @property
    def precatk_vals(self):
        return self.getlist('REPORT', 'precatk_vals', int)

    # @property
    # def report_train_scores(self):
    #    return self._config.getboolean('REPORT', 'report_train_scores')


class Evaluator(object):
    r"""
    Provides functions to evaluate network embedding methods on LP tasks.
    Can evaluate methods from their node embeddings, edge embeddings or edge predictions / similarity scores.

    Parameters
    ----------
    dim : int, optional
        The number of dimensions of the embedding. Default is 128.
    lp_model : object, optional
        The link prediction method to be used. Default is logistic regression.
    """

    def __init__(self, dim=128, lp_model=LogisticRegression(solver='liblinear')):
        # General evaluation parameters
        self.dim = dim
        self.edge_embed_method = None
        self.lp_model = lp_model
        # Train and validation data split objects
        self.traintest_split = split.EvalSplit()
        # Results
        self._results = list()

    def get_results(self):
        r"""
        Returns the scoresheet list
        """
        return self._results

    def reset_results(self):
        r"""
        Resets the scoresheet list
        """
        self._results = list()


    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, verbose=True):
        r"""
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, edge embedding or end to end embedding methods.
        If model parameter tuning is required, this method automatically generates train/validation splits
        with the same parameters as the train/test splits.

        Parameters
        ----------
        method_name : basestring
            A string indicating the name of the method to be evaluated.
        method_type : basestring
            A string indicating the type of embedding method (i.e. ne, ee, e2e)
        command : basestring
            A string containing the call to the method as it would be written in the command line.
            For 'ne' methods placeholders (i.e. {}) need to be provided for the parameters: input network file,
            output file and embedding dimensionality, precisely IN THIS ORDER.
            For 'ee' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train embeddings, output test embeddings and embedding dimensionality, 6 placeholders (i.e. {}) need to
            be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output embeddings, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
            For 'e2e' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train predictions, output test predictions and embedding dimensionality, 6 placeholders (i.e. {}) need
            to be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output predictions, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
        edge_embedding_methods : array-like
            A list of methods used to compute edge embeddings from the node embeddings output by the NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
            When evaluating 'ee' or 'e2e' methods, this parameter is ignored.
        input_delim : basestring
            The delimiter expected by the method as input (edgelist).
        output_delim : basestring
            The delimiter provided by the method in the output
        tune_params : basestring
            A string containing all the parameters to be tuned and their values.
        maximize : basestring
            The score to maximize while performing parameter tuning.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        verbose : bool
            A parameter to control the amount of screen output.
        """
        # If the method evaluated does not require edge embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None


        else:
            # No parameter tuning is needed
            # Call the corresponding evaluation method
            if method_type == 'ne':
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command, edge_embedding_methods,
                                                input_delim, output_delim, write_weights, write_dir, verbose)

            else:
                raise ValueError('Method type {} unknown!'.format(method_type))

            # Store the evaluation results
            self._results.extend(results)

    def _evaluate_ne_cmd(self, data_split, method_name, command, edge_embedding_methods, input_delim, output_delim,
                         write_weights, write_dir, verbose):
        """
        The actual implementation of the node embedding evaluation. Stores the train graph as an edgelist to a
        temporal file and provides it as input to the method evaluated. Performs the command line call and reads
        the output. Node embeddings are transformed to edge embeddings and predictions are run.

        Returns
        -------
        results : list
            A list of results, one for each edge embedding method set.
        """


        # Read the embeddings
        #X = np.genfromtxt(tmpemb, delimiter=output_delim, dtype=float, skip_header=emb_skiprows, autostrip=True)

        X = np.genfromtxt('embedding_8.csv', delimiter=',', dtype=float,autostrip=True)
        #print (X.shape)
        #print (X)


        if X.ndim == 1:
            raise ValueError('Error encountered while reading node embeddings for method {}. '
                             'Please check the output delimiter for the method, this value is probably incorrect.'
                             .format(method_name))

        if X.shape[1] == self.dim:
            # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where rows correspond to sorted node id
            keys = map(str, sorted(data_split.TG.nodes))
            # keys = map(str, range(len(X)))
            X = dict(zip(keys, X))
        elif X.shape[1] == self.dim + 1:
            warnings.warn('Output provided by method {} contains {} columns, {} expected!'
                          '\nAssuming first column to be the nodeID...'
                          .format(method_name, X.shape[1], self.dim), Warning)
            # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
            #keys = map(str, np.array(X[:, 0], dtype=int))
            keys = map(str, range(len(X)))
            X = dict(zip(keys, X[:, 1:]))
        else:
            raise ValueError('Incorrect node embedding dimensions for method {}!'
                             '\nValues expected: {} or {} \nValue received: {}'
                             .format(method_name, self.dim, self.dim + 1, X.shape[1]))





        X_test = np.genfromtxt('embedding_9.csv', delimiter=',', dtype=float, autostrip=True)
        # print(X_text.shape)
        if X_test.ndim == 1:
            raise ValueError('Error encountered while reading node embeddings for method {}. '
                             'Please check the output delimiter for the method, this value is probably incorrect.'
                             .format(method_name))

        if X_test.shape[1] == self.dim:
            # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where rows correspond to sorted node id
            # keys = map(str, sorted(data_split.TG.nodes))
            keys = map(str, range(len(X_test)))
            X_test = dict(zip(keys, X_test))
        elif X_test.shape[1] == self.dim + 1:
            warnings.warn('Output provided by method {} contains {} columns, {} expected!'
                          '\nAssuming first column to be the nodeID...'
                          .format(method_name, X_test.shape[1], self.dim), Warning)
            # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
            #keys = map(str, np.array(X_test[:, 0], dtype=int))
            keys = map(str, range(len(X)))
            X_test = dict(zip(keys, X_test[:, 1:]))
        else:
            raise ValueError('Incorrect node embedding dimensions for method {}!'
                             '\nValues expected: {} or {} \nValue received: {}'
                             .format(method_name, self.dim, self.dim + 1, X_test.shape[1]))




        # Evaluate the model
        results = list()
        for ee in edge_embedding_methods:
            results.append(self.evaluate_ne(data_split=data_split, X=X, method=method_name, X_test=X_test, edge_embed_method=ee))
        return results





    def evaluate_ne(self, data_split, X, method, X_test, edge_embed_method,
                    label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        r"""
        Runs the complete pipeline, from node embeddings to edge embeddings and returns the prediction results.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type str and the values of type array.
        method : basestring
            A string indicating the name of the method to be evaluated.
        edge_embed_method : basestring
            A string indicating the method used to compute edge embeddings from node embeddings.
            The accepted values are any of the function names in evalne.evaluation.edge_embeddings.
        label_binarizer : string or Sklearn binary classifier, optional
            If the predictions returned by the model are not binary, this parameter indicates how these binary
            predictions should be computed in order to be able to provide metrics such as the confusion matrix.
            Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
            as binarization thresholds.
            Default is LogisticRegression(solver='liblinear')
        params : dict
            A dictionary of parameters : values to be added to the results class.

        Returns
        -------
        results : Results
            A results object
        """
        # Run the evaluation pipeline


        tr_edge_embeds, te_edge_embeds = self.compute_ee(data_split, X, edge_embed_method, X_test)

        train_pred, test_pred = self.compute_pred(data_split, tr_edge_embeds, te_edge_embeds)

        return self.compute_results(data_split=data_split, method_name=method, train_pred=train_pred,
                                    test_pred=test_pred, label_binarizer=label_binarizer, params=params)

    def compute_ee(self, data_split, X, edge_embed_method,X_test):
        r"""
        Computes edge embeddings using the given node embeddings dictionary and edge embedding method.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type str and the values of type array.
        edge_embed_method : basestring
            A string indicating the method used to compute edge embeddings from node embeddings.
            The accepted values are any of the function names in evalne.evaluation.edge_embeddings.

        Returns
        -------
        tr_edge_embeds : matrix
            A Numpy matrix containing the train edge embeddings.
        te_edge_embeds : matrix
            A Numpy matrix containing the test edge embeddings.
        """
        self.edge_embed_method = edge_embed_method


        # data_test=np.zeros((10000,2))
        #
        # load_data = open('data1_delta.txt', 'r')
        #
        # raw_data = load_data.read()
        # rows = raw_data.splitlines()
        #
        #
        # for i,row in enumerate(rows):
        #     need = np.fromstring(row, dtype=int, sep=',')
        #
        #     data_test[i][0]=need[0]
        #     data_test[i][1]=need[1]

        data_test=np.load('save_test9_delta.npy')

        #print (data_test)
        #print (data_split.train_edges)


        try:
            func = getattr(edge_embeddings, str(edge_embed_method))
            tr_edge_embeds = func(X, data_split.train_edges)
            #te_edge_embeds = func(X_test, data_test)
            te_edge_embeds = func(X, data_test)
        except AttributeError as e:
            raise AttributeError('Option is not one of the available edge embedding methods! {}'.format(e.message))

        return tr_edge_embeds, te_edge_embeds

    def compute_pred(self, data_split, tr_edge_embeds, te_edge_embeds):
        r"""
        Computes predictions from the given edge embeddings.
        Trains an LP model with the train edge embeddings and performs predictions for train and test edge embeddings.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        tr_edge_embeds : matrix
            A Numpy matrix containing the train edge embeddings.
        te_edge_embeds : matrix
            A Numpy matrix containing the test edge embeddings.

        Returns
        -------
        train_pred : array
            The link predictions for the train data.
        test_pred : array
            The link predictions for the test data.
        """
        # Train the LP model
        self.lp_model.fit(tr_edge_embeds, data_split.train_labels)

        # Predict
        train_pred = self.lp_model.predict_proba(tr_edge_embeds)[:, 1]
        test_pred = self.lp_model.predict_proba(te_edge_embeds)[:, 1]

        return train_pred, test_pred

    def compute_results(self, data_split, method_name, train_pred, test_pred,
                        label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        r"""
        Generates results from the given predictions and returns them.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        method_name : basestring
            A string indicating the name of the method for which the results will be created.
        train_pred :
            The link predictions for the train data.
        test_pred :
            The link predictions for the test data.
        label_binarizer : string or Sklearn binary classifier, optional
            If the predictions returned by the model are not binary, this parameter indicates how these binary
            predictions should be computed in order to be able to provide metrics such as the confusion matrix.
            Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
            as binarization thresholds.
            Default is LogisticRegression(solver='liblinear')
        params : dict, optional
            A dictionary of parameters : values to be added to the results class.
            Default is None.

        Returns
        -------
        results : Results
            Returns the evaluation results.
        """

        # Get global parameters
        if self.edge_embed_method is not None:
            parameters = {'dim': self.dim, 'edge_embed_method': self.edge_embed_method}
        else:
            parameters = {'dim': self.dim}

        # Get data related parameters
        parameters.update(self.traintest_split.get_parameters())

        # Obtain the evaluation parameters
        if params is not None:
            parameters.update(params)

        #np.save('save_test9_delta',data_split.train_edges)
        #np.save('save_test9_label_delta',data_split.train_labels)

        test_labels = np.load('save_test9_label_delta.npy')

        results = score.Results(method=method_name, params=parameters,
                                train_pred=train_pred, train_labels=data_split.train_labels,
                                test_pred=test_pred, test_labels=test_labels,
                                label_binarizer=label_binarizer)

        return results
