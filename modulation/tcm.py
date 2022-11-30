import numpy as np
import commpy.channelcoding.convcode as cc


class TCM:
    def __init__(self):
        # from numpy import sin
        # from numpy import cos
        # from numpy import pi
        # self.constellation = np.array([1 + 1j * 0,
        #                                cos(3 * pi / 4) + 1j * sin(3 * pi / 4),
        #                                cos(1 * pi / 4) + 1j * sin(1 * pi / 4),
        #                                cos(2 * pi / 4) + 1j * sin(2 * pi / 4),
        #                                cos(4 * pi / 4) + 1j * sin(4 * pi / 4),
        #                                cos(7 * pi / 4) + 1j * sin(7 * pi / 4),
        #                                cos(5 * pi / 4) + 1j * sin(5 * pi / 4),
        #                                cos(6 * pi / 4) + 1j * sin(6 * pi / 4)])
        self.constellation = np.array([7, 1, 5, 3, -1, -7, -3, -5])

        self.trellis = cc.Trellis(memory=np.array([2]), g_matrix=np.array([[7, 5]]))
        self.trellis.next_state_table = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]])
        # Таблица выходов. Выходные битовые последовательности длины 3 представлены числами в decimal
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
        return self.constellation[values]

    def decode(self, r):
        decoded_bits = np.full(2 * len(r), -1)
        decoded_bits_num = 0
        self.state_metrics[0][0] = 0
        for i in range(len(r)):
            # считаем метрики переходов в текущем столбце
            transitions = self.get_outgoing_transitions_from_set(self.start_states)
            calc_metrics_for_transitions(transitions, r[i])
            self.save_transitions_to_transition_table(transitions)
            # подсчет метрики каждого состояния
            for state in range(self.trellis.number_states):
                filtered = get_transitions_going_to(state, transitions)
                state_metrica = self.get_min_branch_metrica(filtered)
                self.state_metrics[state][1] = state_metrica

            # делаем traceback и удаляем лишние переходы + декодируем символы, если в столбце остался лишь один переход
            column_num = self.curr_column
            symbols_decoded_per_iteration = 0
            while True:
                useless_states = self.get_useless_states(column_num)
                if len(useless_states) == 0:
                    # начинаем проверку на наличие единственного перехода. если он единственный, то можем декодировать биты
                    # продолжаем процедуру двигаясь слева направо, пока не станет больше 1 перехода в столбце
                    while True:
                        transitions = self.get_all_transitions_at_column(column_num)
                        if len(transitions) == 1:
                            transition = transitions.pop()
                            decoded_symbol = transition.survived_symbol
                            output_symbols = list(self.output_symbols_table[transition.from_state])

                            information_bits = output_symbols.index(decoded_symbol)
                            bit_array = decimal_to_bit_array(information_bits, self.trellis.k)
                            decoded_bits[decoded_bits_num: decoded_bits_num + self.trellis.k] = bit_array
                            decoded_bits_num = decoded_bits_num + self.trellis.k

                            column_num = column_num + 1
                            symbols_decoded_per_iteration = symbols_decoded_per_iteration + 1
                        else:
                            break
                    break
                self.remove_transitions_going_to(useless_states, column_num - 1)
                column_num = column_num - 1

            # обновляем список стартовых состояний
            self.update_start_states(self.get_all_transitions_at_column(self.curr_column))

            # сдвигаем влево таблицу переходов shift_cnt раз
            self.shift_transition_table_to_left_n_times(symbols_decoded_per_iteration)
            self.curr_column = self.curr_column - symbols_decoded_per_iteration

            # сдвигаем влево таблицу метрик состояний
            self.state_metrics[:, 0] = self.state_metrics[:, 1]

            # в таблице переходов переключились на следующий столбик
            self.curr_column = self.curr_column + 1

        return decoded_bits

    def shift_transition_table_to_left_n_times(self, n):
        for j in range(n):
            for i in range(self.tb_depth - 1):
                self.transition_table[:, i] = self.transition_table[:, i + 1]

    def get_transitions_number_at_column(self, column_num):
        cnt = 0
        for i in range(8):
            curr_tr = self.transition_table[i][column_num]
            if curr_tr is not None:
                cnt = cnt + 1
        return cnt

    def get_all_transitions_at_column(self, column_num):
        transitions = []
        for i in range(8):
            curr_tr = self.transition_table[i][column_num]
            if curr_tr is not None:
                transitions.append(curr_tr)
        return transitions

    def remove_transitions_going_to(self, to_states, column_num):
        """в текущем столбце удаляем переходы идущие в указанные состояния, которые находятся справа"""
        for to_state in to_states:
            self.__remove_transitions_going_to(to_state, column_num)

    def __remove_transitions_going_to(self, to_state, column_num):
        """в текущем столбце удаляем переходы идущие в указанное состояние, которое находится справа"""
        for i in range(8):
            curr_tr = self.transition_table[i][column_num]
            if curr_tr is not None and curr_tr.to_state == to_state:
                self.transition_table[i][column_num] = None

    def get_useless_states(self, curr_column_num):
        no_outgoing = self.get_states_without_outgoing_transitions(curr_column_num)
        with_incoming = self.get_states_with_incoming_transitions(curr_column_num - 1)
        useless_states = np.intersect1d(with_incoming, no_outgoing)
        return list(useless_states)

    def get_states_with_incoming_transitions(self, column_num):
        with_incoming_states = []
        for i in range(8):
            curr_tr = self.transition_table[i][column_num]
            if curr_tr is not None:
                with_incoming_states.append(curr_tr.to_state)
        return list(np.unique(with_incoming_states))

    def get_states_without_outgoing_transitions(self, column_num):
        from_states = []
        for i in range(8):
            curr_tr = self.transition_table[i][column_num]
            if curr_tr is not None:
                from_states.append(curr_tr.from_state)
        no_outgoing_states = []
        for i in range(4):
            if from_states.count(i) == 0:
                no_outgoing_states.append(i)
        return no_outgoing_states

    def get_min_branch_metrica(self, filtered_transitions):
        min_metrica = np.inf
        index = -1
        for i in range(len(filtered_transitions)):
            transition = filtered_transitions[i]
            from_state_metrica = self.get_state_metrica(transition.from_state)
            branch_metrica = from_state_metrica + transition.metrica ** 2
            if branch_metrica < min_metrica:
                min_metrica = branch_metrica
                index = i

        # убиваем переход с большей меткой
        if len(filtered_transitions) > 1:
            # верно ли, что в таком случае входящих переходов будет всегда 2?
            k = 1 - index
            transition_to_be_killed = filtered_transitions[k]
            for j in range(8):
                curr = self.transition_table[j][self.curr_column]
                if curr == transition_to_be_killed:
                    self.transition_table[j][self.curr_column] = None

        return min_metrica

    def extract_metrics_from_transition_table(self):
        metrics = np.full((8, self.tb_depth), np.inf)
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


class Transition:
    def __init__(self, from_state: int, to_state: int, output_symbols):
        self.from_state = from_state
        self.to_state = to_state
        self.output_symbols = output_symbols
        self.survived_symbol = None
        self.metrica = np.inf
        self.bits = np.full(2, -1)

    def calc_metrica(self, received_symbol):
        metrics = np.abs(self.output_symbols - received_symbol)
        self.survived_symbol = self.output_symbols[np.argmin(metrics)]
        self.metrica = np.min(metrics)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.from_state == other.from_state and self.to_state == other.to_state
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def print(self):
        print("%i-[%i]->%i" % (self.from_state, self.survived_symbol, self.to_state))


def get_transitions_going_to(to_state, transitions):
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


def decimal_to_bit_array(decimal, arr_length):
    s = np.binary_repr(decimal, arr_length)
    bit_array = string_to_bit_array(s, arr_length)
    return bit_array


def string_to_bit_array(s, arr_length):
    bits = np.zeros(arr_length)
    str_length = len(s)
    if str_length > arr_length:
        raise ValueError("array is too short for that string!")
    else:
        offset = arr_length - str_length
        for i in range(offset, arr_length, 1):
            bits[i] = s[i - offset]
    return bits


def print_transitions(transitions):
    for transition in transitions:
        transition.print()
