import commpy.channelcoding as cc

from interface import BlockConvolutionalCoder, BlockConvolutionalDecoder


class ConvCoder(BlockConvolutionalCoder):

    def get_code_rate(self) -> float:
        return self.trellis.k / self.trellis.n

    def encode(self, input_bits):
        return cc.conv_encode(input_bits, self.trellis)


class ConvDecoder(BlockConvolutionalDecoder):

    def decode(self, data):
        if self.mode == 'hard':
            return cc.viterbi_decode(data, self.trellis)
        elif self.mode == 'unquantized':
            return cc.viterbi_decode(data, self.trellis, decoding_type='unquantized')
        else:
            raise Exception("decoder mode must be 'hard' or 'unquantized'")
