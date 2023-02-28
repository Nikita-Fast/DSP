
from gui_wrappers.block import CBlock
from qam_modulation import QAMModulator


class QAMModulatorGUI(CBlock):
    block: QAMModulator

    def __init__(self, parent=None):
        super().__init__(num_input_ports=1, num_output_ports=1, parent=parent)
        self.block = QAMModulator(bits_per_symbol=2)

    @staticmethod
    def who_i_am():
        return 'QAM Modulator'


