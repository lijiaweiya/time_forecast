import torch
from torch import nn
from layers.Embed import DataEmbedding


class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=10, num_heads=4):
        super().__init__()
        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 分类预测头
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        # 残差连接
        self.residual = nn.Linear(input_dim, hidden_dim)

        # 中心点归一化
        self.center_norm = nn.LayerNorm(hidden_dim)

    def forward(self, centers):
        """处理中心点特征并输出分类分数"""
        # 原始特征转换 [16, 10, 7] -> [16, 10, hidden_dim]
        residual = self.residual(centers)

        # 特征增强 [16, 10, 7] -> [16, 10, hidden_dim]
        enhanced = self.feature_enhancer(centers)

        # 残差连接
        features = enhanced + residual

        # 自注意力机制 [16, 10, hidden_dim] -> [16, 10, hidden_dim]
        attn_output, _ = self.attention(features, features, features)
        features = features + attn_output  # 残差连接

        # 中心点归一化
        features = self.center_norm(features)

        # 分类预测 [16, 10, hidden_dim] -> [16, 10, 1]
        scores = self.classifier_head(features)

        return torch.softmax(scores.squeeze(-1),dim=-1)  # [16, 10]

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

        # Bi-GRU编码器
        self.class_encoder_gru = nn.GRU(
            input_size=configs.d_model,
            hidden_size=configs.d_model // 2,  # 双向GRU需要除以2
            num_layers=configs.e_layers,
            batch_first=True,
            bidirectional=True
        )

        # Bi-GRU解码器
        self.class_decoder_gru = nn.GRU(
            input_size=configs.d_model,
            hidden_size=configs.d_model // 2,  # 双向GRU需要除以2
            num_layers=configs.d_layers,
            batch_first=True,
            bidirectional=True
        )
        # 分类器
        self.next_step_classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),  # 降维
            nn.ReLU(),  # 非线性激活
            nn.Dropout(configs.dropout),  # 防止过拟合
            nn.Linear(configs.d_model // 2, 10)  # 输出层，假设有10个类别
        )
        self.classifier = EnhancedClassifier()

    def foreacat(self, x, x_mark,x_dec, x_mark_dec):
        # 嵌入层
        enc_input = self.embedding(x, None)
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

    def classification(self, x_enc, x_mark, x_dec, x_label):
        """
        x: 输入特征 [16, 96, 7]
        x_label: 标签 [16, 96] (值范围0-9)
        输出: 分类分数 [16, 10]
        """
        device = x_enc.device
        batch_size, seq_len, feat_dim = x_enc.shape
        num_classes = 10

        # 初始化中心点张量 [16, 10, 7]
        centers = torch.zeros(batch_size, num_classes, feat_dim, device=device)
        counts = torch.zeros(batch_size, num_classes, device=device)

        # 向量化计算中心点
        for cls in range(num_classes):
            # 创建当前类别的布尔掩码 [16, 96]
            mask = (x_label == cls)

            # 计算当前类别的样本数 [16]
            cls_count = mask.sum(dim=1).float()  # 转换为浮点数以便除法

            # 避免除零错误 - 将计数为0的替换为1
            cls_count = torch.where(cls_count == 0, torch.ones_like(cls_count), cls_count)

            # 扩展掩码维度以匹配特征 [16, 96, 1]
            mask_exp = mask.unsqueeze(-1).float()

            # 计算当前类别的特征总和 [16, 7]
            cls_sum = (x_enc * mask_exp).sum(dim=1)

            # 计算当前类别的中心点 [16, 7]
            cls_center = cls_sum / cls_count.unsqueeze(-1)

            # 存储结果 - 修复维度问题
            centers[:, cls, :] = cls_center


        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)
        # 嵌入层
        enc_input = self.embedding(x_enc, x_mark)

        # 编码层
        memory = self.encoder(enc_input)
        dec_input = x_enc[:, -self.configs.label_len:, :]  # 使用历史数据的最后若干步作为解码器输入
        dec_input = self.embedding(dec_input, None)
        # 解码层
        output = self.decoder(dec_input, memory)
        # 分类器
        output = self.next_step_classifier(output) # [batch, time_steps, num_classes]
        # 返回第一个时间步的标签预测值
        logits = output[:, 0, :]
        logits = torch.softmax(logits, dim=-1)
        # 通过增强分类器获取分数
        return torch.softmax(self.classifier(centers)+logits,dim=-1)
        # return logits

    def forward(self, x, x_mark,x_dec, x_mark_dec):
        if self.configs.task_name=='long_term_forecast':
            return self.classification(x,x_mark,x_dec,x_mark_dec)
            if self.configs.err:
                return self.err_forecast( x, x_mark,x_dec, x_mark_dec)
            else:
                return self.foreacat( x, x_mark,x_dec, x_mark_dec)


