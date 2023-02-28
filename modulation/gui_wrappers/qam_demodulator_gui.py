
from gui_wrappers.block import CBlock
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator


class QAMDemodulatorGUI(CBlock):
    block: QAMDemodulator

    def __init__(self, parent=None):
        super().__init__(1, 1, parent)
        self.block = QAMDemodulator.from_qam_modulator(QAMModulator(2))

    @staticmethod
    def who_i_am():
        return 'QAM Demodulator'
