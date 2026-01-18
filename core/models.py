# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
import copy


class GraphGRUCell(nn.Module):
    """Graph-based Gated Recurrent Unit cell with relation-aware propagation."""
    
    def __init__(self, in_dim, out_dim, dropout=0.0, num_relations=3, 
                 num_bases=3, K=1, num_nodes=80, global_fusion=False):
        super(GraphGRUCell, self).__init__()
        self.num_chunks = 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_nodes = num_nodes
        self.global_fusion = global_fusion
        
        self.fc_input = nn.Linear(in_dim, out_dim * self.num_chunks)
        self.fc_hidden = nn.Linear(out_dim, out_dim * self.num_chunks)
        
        self.bias_input = Parameter(torch.Tensor(self.out_dim))
        self.bias_reset = Parameter(torch.Tensor(self.out_dim))
        self.bias_new = Parameter(torch.Tensor(self.out_dim))
        self.dropout = dropout
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.ones_(self.bias_input)
        init.ones_(self.bias_reset)
        init.ones_(self.bias_new)
        
        if self.global_fusion:
            init.ones_(self.bias_input)
            init.ones_(self.bias_reset)
            init.ones_(self.bias_new)
    
    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = torch.zeros(inputs.size(0), inputs.size(1), 
                               self.out_dim, dtype=inputs.dtype, 
                               device=inputs.device)
        
        gi = self.fc_input(inputs)
        gh = self.fc_hidden(hidden)
        
        i_input, i_new = gi.chunk(2, 2)
        h_input, h_new = gh.chunk(2, 2)
        
        input_gate = torch.sigmoid(i_input + h_input + self.bias_input)
        new_gate = torch.tanh(i_new + h_new + self.bias_new)
        next_hidden = input_gate * new_gate + (1 - new_gate) * hidden
        
        return next_hidden


class OriginDestinationNet(nn.Module):
    """OD network with encoder-decoder architecture."""
    
    def __init__(self):
        super(OriginDestinationNet, self).__init__()
        
        self.num_nodes = 80
        self.num_output_dim = 80
        self.num_units = 96
        self.num_finished_input_dim = 80
        self.num_unfinished_input_dim = 80
        self.num_rnn_layers = 2
        self.seq_len = 6
        self.horizon = 6
        self.num_relations = 1
        self.K = 2
        self.num_bases = 1
        
        self.dropout_type = None
        self.dropout_prob = 0.0
        self.global_fusion = False
        
        self.encoder_first_finished_cells = GraphGRUCell(
            self.num_finished_input_dim, self.num_units, 
            self.dropout_prob, self.num_relations, 
            num_bases=self.num_bases, K=self.K, 
            num_nodes=self.num_nodes, global_fusion=self.global_fusion)
        
        self.encoder_second_cells = nn.ModuleList([
            GraphGRUCell(self.num_units, self.num_units, 
                        self.dropout_prob, self.num_relations, 
                        num_bases=self.num_bases, K=self.K, 
                        num_nodes=self.num_nodes, 
                        global_fusion=self.global_fusion)
            for _ in range(self.num_rnn_layers - 1)])
        
        self.decoder_first_cells = GraphGRUCell(
            self.num_finished_input_dim, self.num_units, 
            self.dropout_prob, self.num_relations, 
            num_bases=self.num_bases, K=self.K, 
            num_nodes=self.num_nodes, global_fusion=self.global_fusion)
        
        self.decoder_second_cells = nn.ModuleList([
            GraphGRUCell(self.num_units, self.num_units, 
                        self.dropout_prob, self.num_relations, 
                        self.K, num_nodes=self.num_nodes, 
                        global_fusion=self.global_fusion)
            for _ in range(self.num_rnn_layers - 1)])
        
        self.output_type = 'fc'
        if self.output_type == 'fc':
            self.output_layer = nn.Linear(self.num_units, self.num_output_dim)
    
    def encoder_first_layer(self, x_od, finished_hidden):
        finished_out = self.encoder_first_finished_cells(
            inputs=x_od, hidden=finished_hidden)
        return finished_out
    
    def encoder_second_layer(self, index, first_out, enc_second_hidden):
        enc_second_out = self.encoder_second_cells[index](
            inputs=first_out, hidden=enc_second_hidden)
        return enc_second_out
    
    def decoder_first_layer(self, decoder_input, dec_first_hidden):
        dec_first_out = self.decoder_first_cells(
            inputs=decoder_input, hidden=dec_first_hidden)
        return dec_first_out
    
    def decoder_second_layer(self, index, decoder_first_out, dec_second_hidden):
        dec_second_out = self.decoder_second_cells[index](
            inputs=decoder_first_out, hidden=dec_second_hidden)
        return dec_second_out


class DualInfoTransformer(nn.Module):
    """Dual-stream information transformer with multi-head attention."""
    
    def __init__(self, h=4, d_nodes=288, d_channel=512, d_model=96):
        super(DualInfoTransformer, self).__init__()
        assert d_model % h == 0
        
        self.d_nodes = d_nodes
        self.d_model = d_model
        self.d_channel = d_channel
        self.d_k = d_channel // h
        self.h = h
        
        self.od_linears = self.clones(
            nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_channel, kernel_size=1),
                nn.PReLU(d_channel),
                nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
                nn.PReLU(d_channel)), 2)
        
        self.od_out_linears = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
            nn.PReLU(d_channel))
        
        self.od_conv = nn.Sequential(
            nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(in_channels=d_channel, out_channels=d_model, kernel_size=1),
            nn.PReLU(d_model))
        
        self.od_out_conv = nn.Sequential(
            nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(in_channels=d_channel, out_channels=d_model, kernel_size=1),
            nn.PReLU(d_model))
    
    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    def attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)
    
    def MultiHeadedAttention(self, hid_od, hid_od_out):
        hid_od = hid_od.permute(0, 2, 1)
        hid_od_out = hid_od_out.permute(0, 2, 1)
        nbatches = hid_od.size(0)
        
        odquery, odvalue = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.od_linears, (hid_od, hid_od))]
        
        od_out_key = self.od_out_linears(hid_od_out).view(
            nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        attn_od = self.od_conv(
            self.attention(query=odquery, key=od_out_key, value=odvalue)
            .transpose(-2, -1).contiguous()
            .view(-1, self.d_channel, self.d_nodes))
        
        attn_od = attn_od.permute(0, 2, 1)
        return attn_od
    
    def forward(self, hidden_states_od, hidden_states_od_out):
        return self.MultiHeadedAttention(hidden_states_od, hidden_states_od_out)


def mean_channels_h(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True)
    return spatial_sum / F.size(3)


def stdv_channels_h(F):
    assert F.dim() == 4
    F_mean = mean_channels_h(F)
    return F_mean


def mean_channels_w(F):
    assert F.dim() == 4
    spatial_sum = F.sum(2, keepdim=True)
    return spatial_sum / F.size(2)


def stdv_channels_w(F):
    assert F.dim() == 4
    F_mean = mean_channels_w(F)
    return F_mean


class DiVAAttention(nn.Module):
    """Directional Variance Attention mechanism."""
    
    def __init__(self, numT):
        super(DiVAAttention, self).__init__()
        
        self.contrast_h = stdv_channels_h
        self.contrast_w = stdv_channels_w
        channel = 256
        reduction = 16
        
        self.f_h = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
        
        self.f_v = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
    
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        c_h = self.contrast_h(x)
        c_w = self.contrast_w(x)
        
        c_h = c_h.permute(0, 2, 3, 1)
        c_h = self.f_h(c_h)
        a_h = c_h.permute(0, 3, 1, 2)
        
        c_w = c_w.permute(0, 2, 3, 1)
        c_w = self.f_v(c_w)
        a_w = c_w.permute(0, 3, 1, 2)
        
        out = identity * a_w * a_h
        return out


class MultiHeadBiLSTMDiVANet(nn.Module):
    """Multi-headed BiLSTM with DiVA attention."""
    
    def __init__(self, numT):
        super(MultiHeadBiLSTMDiVANet, self).__init__()
        
        self.DiVA_attention = DiVAAttention(numT)
        D = 256
        self.with_horizontal = True
        self.rnn_h = nn.LSTM(numT, D, num_layers=1, batch_first=True, 
                            bias=True, bidirectional=True)
        self.fc = nn.Linear(2 * D, 256)
    
    def forward(self, x):
        B, H, W, C = x.shape
        
        if self.with_horizontal:
            h1 = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h1)
            h = h.reshape(B, H, W, -1)
        
        x = self.fc(h)
        x = x.permute(0, 3, 1, 2)
        
        x_bilstm = self.DiVA_attention(x)
        next_hidden = x_bilstm.permute(0, 2, 3, 1)
        
        return next_hidden


class OutputLayer(nn.Module):
    """Output projection layer with convolutions."""
    
    def __init__(self, numT):
        super(OutputLayer, self).__init__()
        
        self.f1 = nn.Linear(256, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, numT)
        self.conv = nn.Conv2d(80, 256, kernel_size=(1, 1), 
                             padding=(0, 0), stride=(1, 1), bias=True)
        k = numT // 6
        self.conv_out = nn.Conv2d(256, 80, kernel_size=(1, k), 
                                 padding=(0, 0), stride=(1, k), bias=True)
    
    def forward(self, x):
        x1 = self.f1(x)
        x2 = self.f2(x1)
        x = self.f3(x2)
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class DeepGraphConvNet(nn.Module):
    """Deep graph convolutional network module."""
    
    def __init__(self, device, numT):
        super(DeepGraphConvNet, self).__init__()
        self.device = device
        self.MultiHeadBiLSTMDiVANet = MultiHeadBiLSTMDiVANet(numT)
        self.output_layer = OutputLayer(numT)
    
    def forward(self, x):
        x = self.MultiHeadBiLSTMDiVANet(x)
        
        nodes = x.shape[0] * 80 * 80
        self_loops = torch.arange(0, nodes, dtype=torch.long).to(self.device)
        edge_index = torch.stack([self_loops, self_loops], dim=0).to(self.device)
        
        x = self.output_layer(x)
        return x


class UpInterpolation(nn.Sequential):
    """Upsampling with interpolation module."""
    
    def __init__(self, conv, scale, n_in, n_out=12, bias=True):
        m = []
        m.append(conv(n_in, scale * scale * n_out, 3, bias=bias))
        m.append(nn.PixelShuffle(scale))
        super(UpInterpolation, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                    padding=(kernel_size // 2), bias=bias)


class TemporalFusion(nn.Module):
    """Temporal feature fusion module."""
    
    def __init__(self, numT_od_out, numT, bias=True, conv=default_conv):
        super(TemporalFusion, self).__init__()
        
        n_feats = 32
        kernel_size = 3
        scale = 1
        
        self.head = conv(numT_od_out, n_feats, kernel_size, bias)
        self.tail = UpInterpolation(conv, scale, n_feats, numT, bias=bias)
    
    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x


class MetroODPredictionNet(nn.Module):
    """Main network for metro OD prediction."""
    
    def __init__(self, device):
        super(MetroODPredictionNet, self).__init__()
        
        self.OD = OriginDestinationNet()
        self.num_nodes = 80
        self.num_output_dim = 80
        self.num_units = 96
        self.num_finished_input_dim = 80
        self.num_unfinished_input_dim = 80
        self.num_rnn_layers = 2
        
        self.seq_len = 6
        self.horizon = 6
        self.head = 4
        self.d_channel = 512
        
        self.use_curriculum_learning = True
        self.cl_decay_steps = torch.FloatTensor(data=[200])
        self.use_input = True
        self.mediate_activation = nn.ReLU(self.num_units)
        
        self.global_step = 0
        
        self.encoder_first_interact = DualInfoTransformer(
            h=self.head, d_nodes=self.num_nodes, 
            d_model=self.num_units, d_channel=self.d_channel)
        
        self.submodule1 = DeepGraphConvNet(device, 12)
        self.submodule2 = DeepGraphConvNet(device, 6)
        self.submodule3 = DeepGraphConvNet(device, 12)
        
        self.timeFusion = TemporalFusion(6, 1)
        self.f2_out = nn.Linear(80, self.num_units)
    
    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        try:
            return k / (k + math.exp(step / k))
        except OverflowError:
            return float('inf')
    
    def encoder_od_od_out(self, X_od, X_od_out):
        B, T, N, _ = X_od.shape
        enc_hiddens_od = [None] * self.num_rnn_layers
        
        for t in range(T):
            x_od = X_od[:, t, :, :]
            x_od_out = X_od_out[:, t+1:t+7, :, :]
            
            encoder_x_od = self.OD.encoder_first_layer(x_od, enc_hiddens_od[0])
            
            x_od_out = self.timeFusion(x_od_out)
            x_od_out = x_od_out.squeeze(1)
            encoder_x_od_out = self.f2_out(x_od_out)
            
            enc_first_interact_info_od = self.encoder_first_interact(
                encoder_x_od, encoder_x_od_out)
            
            enc_hiddens_od[0] = encoder_x_od + enc_first_interact_info_od
            enc_mid_out_od = encoder_x_od + enc_first_interact_info_od
            
            for index in range(self.num_rnn_layers - 1):
                enc_mid_out_od = self.mediate_activation(enc_mid_out_od)
                enc_mid_out_od = self.OD.encoder_second_layer(
                    index, enc_mid_out_od, enc_hiddens_od[index + 1])
                enc_hiddens_od[index + 1] = enc_mid_out_od
        
        return enc_hiddens_od
    
    def scheduled_sampling(self, out, label, GO):
        use_truth_sequence = False
        
        if use_truth_sequence:
            decoder_input = label
        else:
            decoder_input = out.detach().view(-1, 80, self.num_output_dim)
        
        if not self.use_input:
            decoder_input = GO.detach()
        
        return decoder_input
    
    def decoder_od_od_out(self, sequences_y, enc_hiddens_od):
        predictions_od = []
        Y_od, Y_od_out = sequences_y[0], sequences_y[1]
        
        GO_od = torch.zeros(enc_hiddens_od[0].size()[0], 
                           enc_hiddens_od[0].size()[1], 
                           self.num_output_dim, 
                           dtype=enc_hiddens_od[0].dtype, 
                           device=enc_hiddens_od[0].device)
        
        dec_input_od = GO_od
        dec_hiddens_od = enc_hiddens_od
        
        for t in range(self.horizon):
            y_od, y_od_out = Y_od[:, t, :, :], Y_od_out[:, t, :, :]
            
            dec_first_out_od = self.OD.decoder_first_layer(
                dec_input_od, dec_hiddens_od[0])
            
            dec_hiddens_od[0] = dec_first_out_od
            dec_mid_out_od = dec_first_out_od
            
            for index in range(self.num_rnn_layers - 1):
                dec_mid_out_od = self.mediate_activation(dec_mid_out_od)
                dec_mid_out_od = self.OD.decoder_second_layer(
                    index, dec_mid_out_od, dec_hiddens_od[index + 1])
                dec_hiddens_od[index + 1] = dec_mid_out_od
            
            dec_mid_out_od = self.OD.output_layer(dec_mid_out_od)
            predictions_od.append(dec_mid_out_od)
            dec_input_od = self.scheduled_sampling(dec_mid_out_od, y_od, GO_od)
        
        if self.training:
            self.global_step += 1
        
        return torch.stack(predictions_od).transpose(0, 1)
    
    def week_model(self, x_od, x_od_out, sequences_y):
        enc_hiddens_od = self.encoder_od_od_out(x_od, x_od_out)
        predictions_od = self.decoder_od_od_out(sequences_y, enc_hiddens_od)
        return predictions_od
    
    def forward(self, sequences, sequences_y):
        x_week_od = sequences[0]
        x_day_od = sequences[1]
        x_hour_od = sequences[2]
        x_week_od_out = sequences[3]
        
        predictions_od_week1 = self.week_model(
            x_week_od[:, -6:, :, :], x_week_od_out[:, -12:, :, :], sequences_y)
        
        x_week_od = x_week_od.permute(0, 2, 3, 1)
        x_day_od = x_day_od.permute(0, 2, 3, 1)
        x_hour_od = x_hour_od.permute(0, 2, 3, 1)
        
        predictions_od_week = self.submodule1(x_week_od)
        predictions_od_day = self.submodule2(x_day_od)
        predictions_od_hour = self.submodule3(x_hour_od)
        
        predictions_od_week = predictions_od_week.permute(0, 3, 1, 2)
        predictions_od_day = predictions_od_day.permute(0, 3, 1, 2)
        predictions_od_hour = predictions_od_hour.permute(0, 3, 1, 2)
        
        predictions_od = (predictions_od_week1 + predictions_od_week + 
                         predictions_od_day + predictions_od_hour)
        
        return predictions_od
