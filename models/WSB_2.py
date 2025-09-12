import torch
from torch import nn
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs=configs
        # 嵌入层
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # 编码层
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=configs.d_model, nhead=configs.n_heads, dim_feedforward=configs.d_ff,
                                       dropout=configs.dropout),
            num_layers=configs.e_layers
        )

        # 解码层
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=configs.d_model, nhead=configs.n_heads, dim_feedforward=configs.d_ff,
                                       dropout=configs.dropout),
            num_layers=configs.d_layers
        )

        # 预测层
        self.projection = nn.Linear(configs.d_model, configs.enc_in)

    def foreacat(self, x, x_mark,x_dec, x_mark_dec):
        # 嵌入层
        enc_input = self.embedding(x, x_mark)

        # 编码层
        memory = self.encoder(enc_input)

        # 准备解码器输入
        dec_input = x[:, -self.configs.label_len:, :]  # 使用历史数据的最后若干步作为解码器输入
        dec_input = self.embedding(dec_input, None)

        # 解码层
        output = self.decoder(dec_input, memory)

        # 预测层
        output = self.projection(output)  # [batch, future_steps, features]
        return output

    def err_forecast(self, x, x_mark,x_dec, x_mark_dec):
        enc_input = self.embedding(x,x_mark)
        # 编码层
        memory = self.encoder(enc_input)

        # 准备解码器输入
        dec_input = x[:, -self.configs.label_len:, :]  # 使用历史数据的最后若干步作为解码器输入
        dec_input = self.embedding(dec_input, None)

        # 解码层
        output = self.decoder(dec_input, memory)

        # 预测层
        output = self.projection(output)  # [batch, future_steps, features]
        return output


    def forward(self, x, x_mark,x_dec, x_mark_dec):
        if self.configs.err:
            return self.err_forecast( x, x_mark,x_dec, x_mark_dec)
        else:
            return self.foreacat( x, x_mark,x_dec, x_mark_dec)

