import torch  # noqa: F401
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer
from speechbrain.nnet.pooling import StatisticsPooling

class SVector(torch.nn.Module):
    def __init__(
        self,
        device="cpu",
        d_input = 30,
        d_model = 512,
        n_layers = 6,
        n_head = 8,
        d_output_FFNN=[512, 1500, 512, 512],
        d_feedforward=1024,
        dropout=0.1,
        n_speakers=1
        ):

        super().__init__()
        self.blocks = nn.ModuleList()

        #GET THE MFCC coefficients are ipmlemented in the yaml file, we use Fbanks for this experiment
        #we dont chunk like in the paper the model may overfit
        #[batch, T, 30]
        
        # FFNN1
        self.FFNN1 = nn.Sequential(
            nn.Linear(d_input, d_output_FFNN[0]),
            nn.ReLU()
        )
        #self.FFNN1 = FeedForward(d_input, d_feedforward, d_output_FFNN[0])
        #self.blocks.append(nn.Sequential(
        #    nn.Linear(d_input, d_output_FFNN[0]),
        #    nn.Dropout(dropout),
        #    nn.ReLU(inplace=True),
        #    #nn.Linear(d_feedforward, d_output_FFNN[0])
        #))

        #whats max of T ie max_len?
        #self.positional_encoding = # PositionalEncoding(d_model, dropout, max_len=5000)

        #transformer needs customization according to paper
        #maybe needs further custimization we will see according to paper
        #https://arxiv.org/pdf/2008.04659.pdf
        #use batchnorm https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm
        self.transformer = nn.TransformerEncoder(
                TransformerEncoderLayer(d_model, n_head),
                num_layers=n_layers,
                #norm= for now use layernorm
        )
        #self.blocks.append(
        #    Encoder(
        #        TransformerLayer(
        #            d_model=d_model,
        #            self_attn=MultiHeadAttention(n_head, d_model, dropout, False),
        #            src_attn=None,
        #            feed_forward=FeedForward(d_model, 2048, d_model),
        #            dropout_prob=dropout,
        #        ),
        #        n_layers,
        #        d_model, 
        #        dropout_ffnn = dropout
        #    ), 
        #)

        # FFNN2
        self.FFNN2 = nn.Sequential(
            nn.Linear(d_model, d_output_FFNN[1]),
            nn.LeakyReLU()
        )

        # Statistical pooling. It converts a tensor of variable length
        # into a fixed-length tensor. The statistical pooling returns the
        # mean and the standard deviation.
        self.pooling = StatisticsPooling()

        # FFN3 | speaker embeddings extracted from the affine part ie s-vectors

        self.FFNN3 = nn.Sequential(
            nn.Linear(d_output_FFNN[1] * 2, d_output_FFNN[2]), 
            nn.ReLU()
        )

    def parameters(self):
        return super().parameters(self)

    def forward(self, x, lens=None, wavs=None):
        x1 = self.FFNN1(x)
        x2 = self.positional_encoding(x1)
        x3 = self.transformer(x2)
        x4 = self.FFNN2(x3)
        x5 = self.pooling(x4)
        x6 = self.FFNN3(x5)

        return x6


        

class Classifier(torch.nn.Module):
    def __init__(
        self,
        d_output_FFNN=[512, 1500, 512, 512],
        n_speakers=28
    ):
        super().__init__()

                #this part should be used in the classifier later
        # FFN4
        self.FFNN4 = nn.Sequential(
            nn.Linear(d_output_FFNN[2], d_output_FFNN[3]),
            nn.ReLU()
        )

        #Simple linear
        self.lin1 = nn.Linear(d_output_FFNN[3], n_speakers)

        self.output = nn.Sequential(
            nn.Flatten(),
        )
        
    def forward(self, x, lens=None, wavs=None):
        x1 = self.FFNN4(x)
        x2 = self.lin1(x1)
        y = self.output(x2)
        return y