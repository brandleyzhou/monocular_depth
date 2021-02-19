from __future__ import absolute_import, division, print_function

from hr_layers import *

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.mobile_encoder = mobile_encoder
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                          + self.num_ch_dec[row]*2*(col-1),
                                                                         output_channel=self.num_ch_dec[row] * 2)
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    self.num_ch_enc[row]+ self.num_ch_enc[row + 1] // 2 +
                    self.num_ch_dec[row]*2*(col-1), self.num_ch_dec[row] * 2)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                            self.num_ch_enc[row], self.num_ch_dec[row + 1])
                else:
                    self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                          + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        if self.mobile_encoder:
            self.convs["dispConvScale0"] = Conv3x3(4, self.num_output_channels)
            self.convs["dispConvScale1"] = Conv3x3(8, self.num_output_channels)
            self.convs["dispConvScale2"] = Conv3x3(24, self.num_output_channels)
            self.convs["dispConvScale3"] = Conv3x3(40, self.num_output_channels)
        else:
            for i in range(4):
                self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        #self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        #self.attention_position = ["31", "22", "13", "04"]
        #self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])
                # low_features=["01" feature["X_00"],
                #               "11" feature["X_10"],
                #               "21" feature["X_20"],
                #               "31" feature["X_30"],
                #               "02" feature["X_00"],feature["X_01"],
                #               "12" feature["X_10"],feature["X_11"]
                #               "22" feature["X_20"],feature["X_21"]
                #               "03" feature["X_00"],feature["X_01"],feature["X_02"]
                #               "13" feature["X_10"],feature["X_11"],feature["X_12"]
                #               "04" feature["X_00"],feature["X_01"],feature["X_02"],feature["03"]

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](features["X_{}{}".format(row + 1, col - 1)]), low_features)
            # feature["X_31"] = self.convs["X_31_attention"](self.convs["X_40"](features["X_40"]))
            # feature["X_22"] = self.convs["X_22_attention"](self.convs["X_31"](features["X_31"]))
            # feature["X_13"] = self.convs["X_22_attention"](self.convs["X_22"](features["X_22"]))
            # feature["X_04"] = self.convs["X_13_attention"](self.convs["X_13"](features["X_13"]))
            
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)
            
            # "01"
            # conv = [self.convs["X_10_conv_0"],self.convs["X_10_conv_1"]] 
            # feature["x_01"] = self.nestConv(conv,features["X_10"],low_feature)
            
            # "11"
            # conv = [self.convs["X_20_conv_0"],self.convs["X_20_conv_1"]] 
            # feature["x_11"] = self.nestConv(conv,features["X_20"]),low_feature)
            
            # "21"
            # conv = [self.convs["X_30_conv_0"],self.convs["X_30_conv_1"]] 
            # feature["x_21"] = self.nestConv(conv,features["X_30"]),low_feature)
            
            # "02"
            # conv = [self.convs["X_11_conv_0"],self.convs["X_11_conv_1"]] 
            # conv.append(self.convs["X_02_downsample"]) 
            # feature["x_02"] = self.nestConv(conv,features["X_11"]),low_faeture)
            
            # "12"
            # conv = [self.convs["X_21_conv_0"],self.convs["X_21_conv_1"]] 
            # conv.append(self.convs["X_12_downsample"]) 
            # feature["x_12"] = self.nestConv(conv,features["X_21"]),low_feature)
            
            # "03"
            # conv = [self.convs["X_12_conv_0"],self.convs["X_12_conv_1"]] 
            # conv.append(self.convs["X_03_downsample"]) 
            # feature["x_03"] = self.nestConv(conv,features["X_12"]),low_feature)
        
        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        #outputs[("disparity", "Scale0")] = self.sigmoid(self.convs["dispConvScale0"](x))
        #outputs[("disparity", "Scale1")] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        #outputs[("disparity", "Scale2")] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        #outputs[("disparity", "Scale3")] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))
        # keep same as the monodepth depth decoder outputs##
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))
        return outputs
        
        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        #outputs[("disparity", "Scale0")] = self.sigmoid(self.convs["dispConvScale0"](x))
        #outputs[("disparity", "Scale1")] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        #outputs[("disparity", "Scale2")] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        #outputs[("disparity", "Scale3")] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))
        # keep same as the monodepth depth decoder outputs##
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))
        return outputs
