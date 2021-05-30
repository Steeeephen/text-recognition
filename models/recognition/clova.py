import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""

These classes are taken from Clova AI's deep text recognition benchmark project
They have been aggregated and simplified

https://github.com/clovaai/deep-text-recognition-benchmark

https://arxiv.org/abs/1904.01906

The project involved building a 4-stage framework for text 
recognition, consisting of:

* Transformation: Normalising the input image
* Feature Extraction: Building feature map of image
* Sequence Modelling: Build sequence using characters from feature map
* Prediction: Predict word from characters

We use the most accurate setup that Clova found:

TPS -> Resnet -> BiLSTM -> Attention

"""


class CLOVA(nn.Module):
    def __init__(self, height=32, width=100):
        super(CLOVA, self).__init__()

        # Transformation
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20,
            I_size=(height, width),
            I_r_size=(height, width),
            I_channel_num=1)

        # FeatureExtraction
        self.FeatureExtraction = ResNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # Sequence modeling
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256

        # Prediction
        self.converter = AttnLabelConverter()

        self.Prediction = Attention(
            self.SequenceModeling_output,
            256,
            len(self.converter.character))

    def build(self):
        self = torch.nn.DataParallel(self).to(device)
        self.load_state_dict(torch.load(reader, map_location=device))
        self.text_recogniser.eval()

    def forward(self, input, text):
        # Transformation stage
        input = self.Transformation(input)

        # Feature extraction stage
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling stage
        contextual_feature = self.SequenceModeling(visual_feature)

        # Prediction stage
        prediction = self.Prediction(
            contextual_feature.contiguous(),
            text,
            batch_max_length=26)

        return prediction


"""

Transformations: TPS

"""


class TPS_SpatialTransformerNetwork(nn.Module):
    # Rectification Network of RARE, namely TPS based STN

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(
            self.F,
            self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0),
            self.I_r_size[0],
            self.I_r_size[1],
            2])

        batch_I_r = F.grid_sample(
            batch_I,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        return batch_I_r


class LocalizationNetwork(nn.Module):
    # Localization Network of RARE

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.I_channel_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.localization_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True))

        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)

        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))

        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))

        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)

        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = (
            torch
            .from_numpy(initial_bias)
            .float()
            .view(-1)
        )

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image
        output:    batch_C_prime : Predicted coordinates of fiducial points
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(features)
        ).view(batch_size, self.F, 2)

        return batch_C_prime


class GridGenerator(nn.Module):
    # Grid Generator of RARE, which produces P_prime by multipling T with P

    def __init__(self, F, I_r_size):
        # Generate P_hat and inv_delta_C for later
        super(GridGenerator, self).__init__()

        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        # for multi-gpu, you need register buffer
        self.register_buffer(
            "inv_delta_C",
            torch.tensor(
                self._build_inv_delta_C(self.F, self.C)).float()
        )

        self.register_buffer(
            "P_hat",
            torch.tensor(
                self._build_P_hat(self.F, self.C, self.P)).float()
        )

    def _build_C(self, F):
        # Return coordinates of fiducial points in I_r; C
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))

        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))

        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)

        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_inv_delta_C(self, F, C):
        # Return inv_delta_C which is needed to calculate T
        hat_C = np.zeros((F, F), dtype=float)

        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r

        np.fill_diagonal(hat_C, 1)

        hat_C = (hat_C ** 2) * np.log(hat_C)

        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height

        P = np.stack(
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]

        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
        C_tile = np.expand_dims(C, axis=0)

        P_diff = P_tile - C_tile

        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))

        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)

        return P_hat

    def build_P_prime(self, batch_C_prime):
        # Generate Grid from batch_C_prime [batch_size x F x 2]
        batch_size = batch_C_prime.size(0)

        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)

        batch_C_prime_with_zeros = torch.cat(
            (
                batch_C_prime,
                torch.zeros(batch_size, 3, 2).float().to(device)
            ),
            dim=1)

        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)

        batch_P_prime = torch.bmm(batch_P_hat, batch_T)

        return batch_P_prime


""" 

Feature Extraction: Resnet

"""


class ResNet_FeatureExtractor(nn.Module):
    # FeatureExtractor of FAN

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(
            input_channel,
            output_channel,
            BasicBlock,
            [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        # 3x3 convolution with padding
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel
        ]

        self.inplanes = int(output_channel / 8)

        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(
            block,
            self.output_channel_block[0],
            layers[0])

        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0)

        self.layer2 = self._make_layer(
            block,
            self.output_channel_block[1],
            layers[1],
            stride=1)

        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1))

        self.layer3 = self._make_layer(
            block,
            self.output_channel_block[2],
            layers[2],
            stride=1)

        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(
            block,
            self.output_channel_block[3],
            layers[3],
            stride=1)

        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


"""

Sequence Modelling: BiLSTM

"""


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)

        return output


"""

Prediction: Attention

"""


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(
            input_size,
            hidden_size,
            num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(
            batch_size,
            onehot_dim
        ).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder.
            text : the text-index of each image.
        output: probability distribution at each step
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1

        output_hiddens = torch.FloatTensor(
            batch_size,
            num_steps,
            self.hidden_size
        ).fill_(0).to(device)

        hidden = (
            torch.FloatTensor(
                batch_size,
                self.hidden_size).fill_(0).to(device),
            torch.FloatTensor(
                batch_size,
                self.hidden_size).fill_(0).to(device)
        )

        targets = torch.LongTensor(batch_size).fill_(0).to(device)
        probs = torch.FloatTensor(
            batch_size,
            num_steps,
            self.num_classes
        ).fill_(0).to(device)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(
                targets,
                onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(
                hidden,
                batch_H,
                char_onehots)

            probs_step = self.generator(hidden[0])
            probs[:, i, :] = probs_step
            _, next_input = probs_step.max(1)
            targets = next_input

        return probs


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)

        e = self.score(
            torch.tanh(batch_H_proj + prev_hidden_proj))

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character='0123456789abcdefghijklmnopqrstuvwxyz'):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder.
        # [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image
            batch_max_length: max length of text label in the batch

        output:
            text : the input of attention decoder
            length : the length of output of attention decoder
        """
        length = [len(s) + 1 for s in text]
        batch_max_length += 1
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)

        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)

        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
