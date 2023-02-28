from typing import Type, NewType
from PySide2.QtWidgets import QApplication

from channel import AWGNChannel
from conv_coder import ConvCoder, ConvDecoder
from gui import MainWindow
import sys
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
from interface import *
import commpy.channelcoding as cc
from model import Model
from utils import ComputationParameters

# trellis = cc.Trellis(np.array([3]), g_matrix=np.array([[7, 5]]))
# coder_7_5 = ConvCoder(trellis)
# decoder_7_5 = ConvDecoder.from_coder(coder_7_5, 'unquantized')
#
# modulator = QAMModulator(bits_per_symbol=4)
# awgnc = AWGNChannel(information_bits_per_symbol=modulator.bits_per_symbol)
# demodulator = QAMDemodulator.from_qam_modulator(modulator, mode='soft')
#
# info_block = InformationBlock()
#
# connections = [Connection(coder_7_5, 0, modulator, 0),
#                Connection(modulator, 0, awgnc, 0),
#                Connection(info_block, 0, awgnc, 1),
#                Connection(awgnc, 0, demodulator, 0),
#                Connection(awgnc, 1, demodulator, 1),
#                Connection(demodulator, 0, decoder_7_5, 0)]
#
# block_configs = {
#     info_block: [MethodCallDescription('get_ebn0_db', inputs=[], outputs=[0])],
#     coder_7_5: [MethodCallDescription('encode', inputs=[0], outputs=[0])],
#     modulator: [MethodCallDescription('process', inputs=[0], outputs=[0])],
#     awgnc: [MethodCallDescription('process', inputs=[0, 1], outputs=[0]),
#             MethodCallDescription('calc_noise_variance', inputs=[1], outputs=[1])],
#     demodulator: [MethodCallDescription('process', inputs=[0, 1], outputs=[0])],
#     decoder_7_5: [MethodCallDescription('decode', inputs=[0], outputs=[0])]
# }
#
# model = Model(blocks=[info_block, modulator, awgnc, demodulator, coder_7_5, decoder_7_5],
#               starting_blocks=[coder_7_5],
#               final_block=decoder_7_5,
#               connections=connections,
#               block_configs=block_configs,
#               info_block=info_block)
#
# p1 = ComputationParameters(2500, 300_000, [0, 1, 2, 3, 4, 5, 6, 7], 50_000)
# res = model.do_modelling(p1)
# res.plot()

app = QApplication([])
window = MainWindow()
sys.exit(app.exec_())
