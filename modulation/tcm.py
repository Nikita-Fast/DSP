import commpy
import numpy as np
import matplotlib.pyplot as plt
import time

import bpsk_modem
import channel
import qam_modulation
import tcm
from channel import AWGNChannel
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
import commpy.channelcoding.convcode as cc


class Transition:
    def __init__(self, from_state :int, to_state :int, output_symbols):
        self.from_state = from_state
        self.to_state = to_state
        self.output_symbols = output_symbols
        self.survived_symbol = -1
        self.metrica = np.inf

    def calc_metrica(self, received_symbol):
        metrics = np.abs(self.output_symbols - received_symbol)
        self.survived_symbol = np.argmin(metrics)
        self.metrica = np.min(metrics)


def filter_transitions(transitions, to_state):
    """
    Возвращает только переходы ведущие в указанное состояние
    """
    filtered = []
    for transition in transitions:
        if transition.to_state == to_state:
            filtered.append(transition)
    return filtered


def calc_metrics_for_transitions(transitions, received_symbol):
    for tr in transitions:
        tr.calc_metrica(received_symbol)


class TCM:
    def __init__(self):
        from numpy import sin
        from numpy import cos
        from numpy import pi
        # self.constellation = np.array([1 + 1j * 0,
        #                                cos(3 * pi / 4) + 1j * sin(3 * pi / 4),
        #                                cos(1 * pi / 4) + 1j * sin(1 * pi / 4),
        #                                cos(2 * pi / 4) + 1j * sin(2 * pi / 4),
        #                                cos(4 * pi / 4) + 1j * sin(4 * pi / 4),
        #                                cos(7 * pi / 4) + 1j * sin(7 * pi / 4),
        #                                cos(5 * pi / 4) + 1j * sin(5 * pi / 4),
        #                                cos(6 * pi / 4) + 1j * sin(6 * pi / 4)])
        self.constellation = np.array([7,1,5,3,-1,-7,-3,-5])

        self.trellis = cc.Trellis(memory=np.array([2]), g_matrix=np.array([[7, 5]]))
        self.trellis.next_state_table = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]])
        self.trellis.output_table = np.array([[0, 3, 4, 7], [2, 1, 6, 5], [3, 0, 7, 4], [1, 2, 5, 6]])
        self.trellis.number_inputs = 4
        self.trellis.k = 2
        self.trellis.n = 3
        self.output_symbols_table = self.constellation[self.trellis.output_table]

        self.start_states = [0]
        self.state_metrics = np.full((4, 2), np.inf)
        self.tb_depth = 15
        self.transition_table = np.empty((8, self.tb_depth), dtype=Transition)
        self.curr_column = 0

    def encode(self, bits):
        coded_bits = cc.conv_encode(bits, self.trellis)
        values = np.empty(len(coded_bits) // 3, dtype=int)
        for i in range(len(coded_bits) // 3):
            c1 = coded_bits[3 * i]
            c2 = coded_bits[3 * i + 1]
            c3 = coded_bits[3 * i + 2]
            value = (c1 << 2) + (c2 << 1) + c3
            values[i] = value
        print("tcm-coder-output: ", values)
        return self.constellation[values]

    def decode(self, r):
        self.state_metrics[0][0] = 0
        for i in range(len(r)):
            # считаем метрики переходов в текущем столбце
            transitions = self.get_outgoing_transitions_from_set(self.start_states)
            calc_metrics_for_transitions(transitions, r[i])
            self.save_transitions_to_transition_table(transitions)
            # подсчет метрики каждого состояния
            for state in range(4):
                filtered = filter_transitions(transitions, state)
                state_metrica = self.get_min_branch_metrica(filtered)
                self.state_metrics[state][1] = state_metrica
            # обновляем список стартовых состояний
            self.update_start_states(transitions)
            # сдвигаем влево таблицу метрик состояний
            self.state_metrics[:,0] = self.state_metrics[:,1]
            # в таблице переходов переключились на следующий столбик
            self.curr_column = self.curr_column + 1

    def extract_metrics_from_transition_table(self):
        metrics = np.full(((8, self.tb_depth)), np.inf)
        for i in range(8):
            for j in range(self.tb_depth):
                transition = self.transition_table[i][j]
                if transition is not None:
                    metrics[i][j] = transition.metrica
        return metrics

    def save_transitions_to_transition_table(self, transitions):
        cnt = 0
        for tr in transitions:
            self.transition_table[cnt][self.curr_column] = tr
            cnt = cnt + 1

    def update_start_states(self, transitions):
        start_states = []
        for tr in transitions:
            start_states.append(tr.to_state)
        start_states = np.unique(start_states)
        self.start_states = start_states

    def get_outgoing_transitions(self, from_state):
        to_states = self.trellis.next_state_table[from_state]
        transitions = []
        for to_state in np.unique(to_states):
            idx = np.argwhere(to_states == to_state)
            idx = np.reshape(idx, (len(idx)))
            symbols = self.output_symbols_table[from_state][idx]
            transition = Transition(from_state, to_state, symbols)
            transitions.append(transition)
        return transitions

    def get_outgoing_transitions_from_set(self, start_states):
        transitions = []
        for state in start_states:
            transitions = transitions + self.get_outgoing_transitions(state)
        return transitions

    def get_state_metrica(self, state_num):
        return self.state_metrics[state_num][0]

    def get_min_branch_metrica(self, filtered_transitions):
        min_metrica = np.inf
        for transition in filtered_transitions:
            from_state_metrica = self.get_state_metrica(transition.from_state)
            branch_metrica = from_state_metrica + transition.metrica
            min_metrica = min(min_metrica, branch_metrica)
        return min_metrica




