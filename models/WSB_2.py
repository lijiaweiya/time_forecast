from layers.Embed import DataEmbedding
import torch.nn.functional as F
import torch
import torch.nn as nn


# 用于训练聚类结果
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 假设 x 是 [batch_size, seq_len, input_dim]
        # 我们可以将时间步长展平
        batch_size, seq_len, input_dim = x.shape
        x = x.reshape(batch_size, seq_len * input_dim)
        # print("22222")
        # print(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# 增强分类器模块，用于误差分类部分
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

        return torch.softmax(scores.squeeze(-1), dim=-1)  # [16, 10]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        # 嵌入层
        #self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

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
            nn.Linear(configs.d_model // 2, 9)  # 输出层，假设有10个类别
        )
        self.classifier = EnhancedClassifier()

        self.bilstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # self.classier1 = ClassificationModel(input_dim=configs.seq_len*configs.d_model, hidden_dim=12, class_num=configs.cluster)
        self.classier1 = AttentionClassificationModel(
            input_dim=256,  # 每个时间步的特征维度
            seq_len=configs.seq_len,  # 序列长度
            hidden_dim=256,  # 隐藏层维度（可以根据需要调整）
            num_heads=4,  # 注意力头数（可以根据需要调整）
            num_layers=2,  # 注意力层数（可以根据需要调整）
            class_num=configs.cluster,  # 类别数量
            dropout=configs.dropout  # Dropout率
        )

    def foreacat(self, x, x_mark, x_dec, x_mark_dec):
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

    def err_forecast(self, x, x_mark, x_dec, x_mark_dec):
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

    def classification1(self, x_enc, x_mark, x_dec, x_mark_dec):
        """
        x: 输入特征 [16, 96, 7]
        输出: 分类分数 [16, class_num]
        """
        enc_input = self.embedding(x_enc, x_mark)
        lstm_out, _ = self.bilstm(enc_input)
        # print(x_enc.shape)
        # print(enc_input.shape)
        # exit()
        return self.classier1(lstm_out)

    def classification(self, x_enc, x_mark, x_dec, x_label):
        """
        x: 输入特征 [16, 96, 7]
        x_label: 标签 [16, 96] (值范围0-9)
        输出: 分类分数 [16, 10]
        """
        device = x_enc.device
        batch_size, seq_len, feat_dim = x_enc.shape
        num_classes = 9

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
        output = self.next_step_classifier(output)  # [batch, time_steps, num_classes]
        # 返回第一个时间步的标签预测值
        logits = output[:, 0, :]
        logits = torch.softmax(logits, dim=-1)
        # 通过增强分类器获取分数
        return torch.softmax(self.classifier(centers) + logits, dim=-1)
        # return logits

    def forward(self, x, x_mark, x_dec, x_mark_dec):
        if self.configs.task_name == 'long_term_forecast':
            return self.classification1(x, x_mark, x_dec, x_mark_dec)
            if self.configs.err:
                return self.err_forecast(x, x_mark, x_dec, x_mark_dec)
            else:
                return self.foreacat(x, x_mark, x_dec, x_mark_dec)


class AttentionClassificationModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=256, num_heads=4, num_layers=2, class_num=10, dropout=0.1):
        super(AttentionClassificationModel, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.class_num = class_num

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # 前馈网络层
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # 层归一化
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, class_num)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape

        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # 添加位置编码
        x = x + self.positional_encoding[:, :seq_len, :]

        # 通过多个注意力层
        for i in range(self.num_layers):
            # 自注意力
            attn_output, attn_weights = self.attention_layers[i](x, x, x)

            # 残差连接和层归一化
            x = self.layer_norms1[i](x + attn_output)

            # 前馈网络
            ff_output = self.feed_forward_layers[i](x)

            # 残差连接和层归一化
            x = self.layer_norms2[i](x + ff_output)

        # 全局平均池化
        x = x.mean(dim=1)  # [batch_size, hidden_dim]

        # 分类头
        x = self.classification_head(x)

        return F.softmax(x, dim=1)
