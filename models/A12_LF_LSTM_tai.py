"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subNets import SubNet, TextSubNet


class A12_LF_LSTM_tai(nn.Module):
    """
    late fusion using DNN
    """

    def __init__(self, args):
        super(A12_LF_LSTM_tai, self).__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims

        self.text_out = args.text_out
        self.audio_out = args.audio_out
        self.video_out = args.video_out

        self.post_fusion_dim = args.post_fusion_dim
        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        self.audio_subnet = TextSubNet(self.audio_in, self.audio_hidden, self.audio_out, dropout=self.audio_prob)
        self.video_subnet = TextSubNet(self.video_in, self.video_hidden, self.video_out, dropout=self.video_prob)

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.audio_out + self.video_out, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x):
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)
        res = {
            'M': output,
        }
        return res
