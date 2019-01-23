from collections import ChainMap
import itertools
import os
import subprocess

import numpy as np

import critfinder.utils.dataframes
import critfinder.utils.load


def run_from_args_list(script, args_list):
    return subprocess.run(["python", script] + args_list,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_script_pair(setup_args, run_args, pair="optimization"):

    setup_script = "setup_{}_experiment.py".format(pair)
    run_script = "run_{}_experiment.py".format(pair)
    setup_cp = run_from_args_list(setup_script, setup_args)

    if setup_cp.returncode > 0:
        return setup_cp, None

    run_cp = run_from_args_list(run_script, run_args)

    return setup_cp, run_cp


def make_optimizer_setup_args(data_path, **kwargs):
    args = ["--data_path", data_path]

    args = add_kwargs(kwargs, args)

    return args


def make_optimizer_run_args(num_steps, **kwargs):
    args = []

    args = add_kwargs(kwargs, args)

    args += [str(num_steps)]

    return args


def make_critfinder_setup_args(optimization_path, finder, **kwargs):
    args = []

    args = add_kwargs(kwargs, args)

    args += [optimization_path, finder]

    return args


def make_critfinder_run_args(critfinder_path, output_path, num_iters, **kwargs):
    args = []

    args = add_kwargs(kwargs, args)

    args += [critfinder_path, output_path, str(num_iters)]

    return args


def add_kwargs(kwargs, args):
    for arg_name, arg_value in kwargs.items():
        if arg_value is not None:
            args += ["--" + arg_name]
            if str(arg_value).strip() != "":
                args += [str(arg_value)]
    return args


def show_cp(cp):
    print("args:\n", cp.args)
    print("stdout:\n", cp.stdout.decode("utf-8"))
    print("stderr:\n", cp.stderr.decode("utf-8"))


def print_results(results):
    for ii, result in enumerate(results):
        print("=" * 10 + "\n" + str(ii) + ":")

        if result[0].returncode > 0:
            print("error  in setup:")
            show_cp(result[0])

        elif result[1].returncode > 0:
            print("error in run:")
            show_cp(result[1])


def match_kwargs_and_df(list_of_kwargs, df):

    for kwargs, (ii, row) in zip(list_of_kwargs, df.iterrows()):

        # regularization
        if "regularizer" in kwargs.keys():
            assert kwargs["regularizer"] == row.regularizer_str

        # network

        if "include_biases" in kwargs.keys():
            if kwargs["include_biases"] is None:
                assert not row.has_biases
            else:
                assert row.has_biases

        if "nonlinearity" in kwargs.keys():
            if kwargs["nonlinearity"] is not None:
                assert kwargs["nonlinearity"] == row.nonlinearity_str

        # optimizer

        if "optimizer" in kwargs.keys():
            if kwargs["optimizer"] is not None:
                assert kwargs["optimizer"] == row.optimizer_str

        if "optimizer_lr" in kwargs.keys():
            if kwargs["optimizer_lr"] is not None:
                assert kwargs["optimizer_lr"] == row.optimizer_kwargs["lr"]

        if "optimizer_momentum" in kwargs.keys():
            if kwargs["optimizer_momentum"] is not None:
                assert kwargs["optimizer_momentum"] == row.optimizer_kwargs["momentum"]


def recreation_test(row, experiment_type="optimization"):

    if experiment_type == "optimization":
        data_json_path, network_json_path = row.data_json, row.network_json,
        optimizer_json_path = row.optimizer_json
        _, _, experiment = critfinder.util.load.from_paths(
            data_json_path, network_json_path, optimizer_json_path)

        thetas = get_thetas(os.path.dirname(data_json_path))

    elif experiment_type == "critfinder":
        experiment_json_path = row.experiment_json
        experiment_json = critfinder.util.load.open_json(experiment_json_path)
        optimization_path = experiment_json["optimization_path"]

        data_json_path = os.path.join(optimization_path, "data.json")
        network_json_path = os.path.join(optimization_path, "network.json")

        finder_json_path = row.finder_json

        _, _, experiment = critfinder.util.load.from_paths(
            data_json_path, network_json_path, finder_json_path,
            experiment_type=experiment_type)

        thetas = get_thetas(os.path.dirname(finder_json_path))

    experiment_output = experiment.run(thetas[0], len(thetas) - 1)

    if experiment_type == "optimization":
        recreated_thetas = experiment_output
    elif experiment_type == "critfinder":
        recreated_thetas = experiment.runs[-1]["theta"]

    if not np.array_equal(thetas, recreated_thetas):
        return (False, [thetas, recreated_thetas])
    else:
        return (True, None)


def get_thetas(optimization_folder):
    thetas = np.load(os.path.join(optimization_folder, "trajectories", "0000.npz"))["theta"]
    return thetas


class KwargTester(object):
    """Iterable of dictionaries suitable for use as kwargs, designed to test a function f
    by calling it with [f(**kwargs) for kwargs in kwarg_tester].

    The iterable is constructed from an iterable of co-dependent keyword argument names,
    called key_blocks, and an iterable (called val_blocks) of iterables (called val_sets)
    of desired combinations of values.

    The result is a product over val_sets organized as an iterable of dictionaries,
    all sharing the same keys and corresponding to different possible combinations of
    a val_set from each val_block.

    The simplest case is that each key_block is an iterable with one string, corresponding to a
    single keyword argument, and each val_block is an iterable containing one iterable with
    values corresponding to desired values for that keyword argument.
    The result is an iterable of dictionaries, corresponding to the set of all combinations of
    desired values for all keyword arguments.

    ```
    def f(foo, bar, baz=None):
        assert baz is None
        print(" ".join([bar]*foo))

    kwt = KwargTester([["foo", "bar"], ["baz"]], [[[1, "a"], [2, "b"]], [[None]]])

    [f(**kwargs) for kwargs in kwt];
    ```

    ```
    a
    b b
    ```
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.kwarg_product = self.make_kwarg_product()

    def make_kwarg_product(self):
        return itertools.product(*[block.dict_list for block in self.blocks])

    def flatten(self, kwargs_list):
        return dict(ChainMap(*kwargs_list))

    def __iter__(self):
        return [self.flatten(kwargs_list) for kwargs_list in self.kwarg_product].__iter__()

    @classmethod
    def from_raw(cls, keys_of_blocks, val_sets_of_blocks):
        blocks = [Block(keys, val_sets) for keys, val_sets
                  in zip(keys_of_blocks, val_sets_of_blocks)]
        return cls(blocks)


class Block(object):
    """A collection of keyword argument names and an iterable of iterables of
    desired combinations of values for those keyword arguments.

    These are combined into a list of dictionaries, block.dict_list.
    Each dictionary in the list has keys from self.keys and values from one
    of the val_sets.
    """

    def __init__(self, keys, val_sets):
        self.keys = keys
        self.val_sets = val_sets

    @property
    def dict_list(self):
        return [{key: val for key, val in zip(self.keys, val_set)} for val_set in self.val_sets]

    def __repr__(self):
        return (self.keys.__repr__(), self.val_sets.__repr__()).__repr__()
