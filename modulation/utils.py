from typing import List

import numpy as np
from matplotlib import pyplot as plt

from interface import Block, Connection


class BERComputationResult:
    def __init__(self, ber_points: List[float], description: str):
        self.ber_points = ber_points
        self.description = description

    def plot(self):
        plt.yscale("log")
        plt.grid(visible='true')
        plt.xlabel("Eb/N0, dB")
        plt.ylabel("BER")

        plt.plot(self.ber_points, '--o', label=self.description)
        plt.legend()
        plt.show()


def plot_ber_computation_results(results: List[BERComputationResult]):
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")

    for res in results:
        plt.plot(res.ber_points, '--o', label=res.description)
        plt.legend()
    plt.show()


class ComputationParameters:
    def __init__(self, errors_threshold: int, max_processed_bits: int, enb0_range,
                 bits_process_per_iteration=10_000):
        self.errors_threshold = errors_threshold
        self.max_processed_bits = max_processed_bits
        self.ebn0_range = enb0_range
        self.bits_process_per_iteration = bits_process_per_iteration


def gen_bits(size):
    return np.random.randint(low=0, high=2, size=size)


def count_errors(arr1, arr2):
    """Если длины сравниваемых массивов неравны, то сравниваются куски длинной равной длине меньшего из массивов"""
    errs = 0
    for i in range(min(len(arr1), len(arr2))):
        if arr1[i] != arr2[i]:
            errs = errs + 1
    return errs


def get_connections_to(to_block: Block, connections: List[Connection]):
    proper_connections = []
    for connection in connections:
        if connection.to_block == to_block:
            proper_connections.append(connection)
    return proper_connections
