from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_


class HAMMER(nn.Module):
    def __init__(self,
                 args=None,  # 可选参数，用于传递额外的配置
                 config=None,  # 模型配置字典
                 text_encoder=None,  # 文本编码器的路径或名称
                 tokenizer=None,  # 分词器，用于文本处理
                 init_deit=True  # 是否初始化视觉变换器的标志
                 ):
        super().__init__()  # 调用父类的构造函数

        self.args = args  # 保存传入的参数
        self.tokenizer = tokenizer  # 保存分词器
        embed_dim = config['embed_dim']  # 获取嵌入维度

        # 初始化视觉编码器（Vision Transformer）
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # 如果需要，初始化视觉变换器的预训练权重
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]  # 获取模型状态字典
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)  # 插值位置嵌入
            state_dict['pos_embed'] = pos_embed_reshaped  # 更新位置嵌入
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)  # 加载状态字典
            print(msg)  # 打印加载信息

        vision_width = config['vision_width']  # 获取视觉宽度
        bert_config = BertConfig.from_json_file(config['bert_config'])  # 从配置文件加载BERT配置

        # 初始化文本编码器（BERT）
        self.text_encoder = BertForTokenClassification.from_pretrained(text_encoder,
                                                                       config=bert_config,
                                                                       label_smoothing=config['label_smoothing'])

        text_width = self.text_encoder.config.hidden_size  # 获取文本编码器的隐藏层维度
        self.vision_proj = nn.Linear(vision_width, embed_dim)  # 视觉特征的投影层
        self.text_proj = nn.Linear(text_width, embed_dim)  # 文本特征的投影层

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])  # 温度参数
        self.queue_size = config['queue_size']  # 队列大小
        self.momentum = config['momentum']  # 动量参数

        # 创建信息传递头（ITM head）
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # 创建边界框头（bbox head）
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # 创建多分类头（multi-cls head）
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # 创建动量模型
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)  # 动量视觉投影层
        self.text_encoder_m = BertForTokenClassification.from_pretrained(text_encoder,
                                                                         config=bert_config,
                                                                         label_smoothing=config['label_smoothing'])
        self.text_proj_m = nn.Linear(text_width, embed_dim)  # 动量文本投影层

        # 存储模型原始和动量版本的配对
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()  # 复制参数到动量模型

        # 创建图像和文本的队列
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))  # 图像特征队列
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))  # 文本特征队列
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 队列指针

        # 对队列进行归一化处理
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr = nn.LayerNorm(text_width)  # 聚合层的层归一化
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))  # 可学习的分类标记
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)  # 多头注意力机制

        self.norm_layer_it_cross_atten = nn.LayerNorm(text_width)  # 创建用于交叉注意力的层归一化，确保输入特征的均值为0，方差为1
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0,
                                                   batch_first=True)  # 创建多头注意力机制，处理文本和视觉特征之间的交互，12为头的数量
        # 对可学习的分类标记进行截断正态分布初始化，标准差为0.02，确保参数初始化在合理范围内
        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)  # 应用自定义的权重初始化方法，初始化模型中所有层的权重

    def _init_weights(self, m):
        """
        初始化模型权重的辅助函数。

        Args:
            m: 传入的模块（层），根据类型进行权重初始化。
        """
        if isinstance(m, nn.Linear):  # 如果模块是线性层
            trunc_normal_(m.weight, std=.02)  # 使用截断正态分布初始化权重，标准差为0.02
            if m.bias is not None:  # 如果存在偏置
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
        elif isinstance(m, nn.LayerNorm):  # 如果模块是层归一化
            nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # 将权重初始化为1.0

    def build_mlp(self, input_dim, output_dim):
        """
        构建一个多层感知机（MLP）模型。

        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度

        Returns:
            nn.Sequential: 由线性层、层归一化和激活函数组成的序列模型
        """
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),  # 第一个线性层，输出维度为输入维度的两倍
            nn.LayerNorm(input_dim * 2),  # 层归一化
            nn.GELU(),  # GELU激活函数
            nn.Linear(input_dim * 2, input_dim * 2),  # 第二个线性层，保持输出维度
            nn.LayerNorm(input_dim * 2),  # 层归一化
            nn.GELU(),  # GELU激活函数
            nn.Linear(input_dim * 2, output_dim)  # 最后一个线性层，输出指定维度
        )

    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        计算边界框损失，包括L1损失和广义IoU损失。

        Args:
            output_coord: 模型预测的边界框坐标
            target_bbox: 目标边界框坐标
            is_image: 可选，指示每个边界框是否为图像边界框的布尔张量

        Returns:
            tuple: 包含L1损失和广义IoU损失的平均值
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # 计算L1损失，返回每个边界框的损失

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)  # 将预测的中心坐标格式转换为(x1, y1, x2, y2)格式
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)  # 将目标边界框转换为相同格式
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # 检查是否存在退化的边界框（例如，宽度或高度为负）
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)  # 如果有退化框，损失设为0
        else:
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # 计算广义IoU损失

        if is_image is None:  # 如果没有提供is_image
            num_boxes = target_bbox.size(0)  # 使用目标边界框的数量
        else:
            num_boxes = torch.sum(1 - is_image)  # 计算有效边界框的数量
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))  # 仅对有效边界框计算L1损失
            loss_giou = loss_giou * (1 - is_image)  # 仅对有效边界框计算IoU损失

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes  # 返回平均L1损失和平均IoU损失

    def forward(self, image, label, text,  alpha=0, is_train=True):
        # fake_image_box, fake_text_pos,
        """
        前向传播函数，用于模型的训练和推理过程。

        Args:
            image: 输入的图像数据
            label: 输入的标签数据
            text: 输入的文本数据
            fake_image_box: 假的图像边界框，用于计算损失
            fake_text_pos: 假的文本位置，用于计算损失
            alpha: 控制对齐目标的平衡参数
            is_train: 布尔值，指示是否处于训练模式

        Returns:
            tuple: 包含多个损失值的元组
        """
        if is_train:  # 如果处于训练模式
            with torch.no_grad():  # 在更新临时变量时不计算梯度
                self.temp.clamp_(0.001, 0.5)  # 限制温度的范围

            ##================= multi-label convert ========================##
            multicls_label, real_label_pos = get_multi_label(label, image)  # 获取多标签和真实标签位置

            ##================= MAC ========================##
            image_embeds = self.visual_encoder(image)  # 使用视觉编码器处理输入图像
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # 创建图像的注意力掩码

            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # 规范化图像特征

            text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                                 return_dict=True, mode='text')  # 使用文本编码器处理输入文本
            text_embeds = text_output.last_hidden_state  # 获取文本的最后隐藏状态
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)  # 规范化文本特征

            # 获取动量特征
            with torch.no_grad():  # 不计算梯度
                self._momentum_update()  # 更新动量
                image_embeds_m = self.visual_encoder_m(image)  # 处理输入图像以获取动量特征
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)  # 规范化动量图像特征
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # 合并动量特征和图像队列

                text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,
                                                         return_dict=True, mode='text')  # 处理文本以获取动量特征
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]),
                                          dim=-1)  # 规范化动量文本特征
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)  # 合并动量特征和文本队列

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # 计算图像到文本的相似度
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # 计算文本到图像的相似度

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)  # 初始化相似度目标
                # 精细对齐：仅原始应对齐，1表示图像-文本对齐
                sim_targets[real_label_pos, real_label_pos] = 1

                sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)  # 初始化全局对齐目标
                sim_targets_g2g.fill_diagonal_(1)  # 填充对角线为1，表示自对齐

                # 计算最终的相似度目标
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp  # 计算当前图像到文本的相似度
            sim_t2i = text_feat @ image_feat_all / self.temp  # 计算当前文本到图像的相似度

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()  # 计算图像到文本的交叉熵损失
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()  # 计算文本到图像的交叉熵损失

            # in-modality g2g loss: 计算同模态的自对齐损失
            sim_i2i = image_feat @ image_feat_all / self.temp  # 计算图像特征之间的相似度
            sim_t2t = text_feat @ text_feat_all / self.temp  # 计算文本特征之间的相似度

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets_g2g, dim=1).mean()  # 计算图像特征自对齐损失
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets_g2g, dim=1).mean()  # 计算文本特征自对齐损失

            loss_MAC = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4  # 综合损失，取四个损失的平均值

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)  # 更新队列，存储动量特征

            ##================= BIC ========================##
            # 前向传播正样本的图像-文本对
            output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                                attention_mask=text.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                mode='fusion',  # 模式设置为融合
                                                )
            with torch.no_grad():  # 不计算梯度
                bs = image.size(0)  # 获取批量大小

            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)  # 初始化图像-文本匹配标签
            itm_labels[real_label_pos] = 0  # 精细匹配：仅原始样本应匹配，0表示图像-文本匹配
            vl_output = self.itm_head(output_pos.last_hidden_state[:, 0, :])  # 通过匹配头获取输出
            loss_BIC = F.cross_entropy(vl_output, itm_labels)  # 计算图像-文本匹配的交叉熵损失

            # ##================= MLC ========================##
            # output_cls = self.cls_head(output_pos.last_hidden_state[:, 0, :])  # 通过分类头获取输出
            # loss_MLC = F.binary_cross_entropy_with_logits(output_cls,
            #                                               multicls_label.type(torch.float))  # 计算多标签分类的二元交叉熵损失

            # ##================= IMG ========================##
            # # 处理视觉部分的局部特征
            # cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)  # 扩展类别标记以匹配批量大小
            #
            # text_attention_mask_clone = text.attention_mask.clone()  # 克隆文本注意力掩码
            # local_feat_padding_mask_text = text_attention_mask_clone == 0  # 创建局部特征填充掩码，0表示填充标记
            #
            # # 计算图像与文本之间的交叉注意力
            # local_feat_it_cross_attn = image_embeds + self.it_cross_attn(
            #     query=self.norm_layer_it_cross_atten(image_embeds),
            #     key=self.norm_layer_it_cross_atten(text_embeds),
            #     value=self.norm_layer_it_cross_atten(text_embeds),
            #     key_padding_mask=local_feat_padding_mask_text
            # )[0]  # 计算局部特征的交叉注意力
            #
            # # 聚合局部特征
            # local_feat_aggr = self.aggregator(
            #     query=self.norm_layer_aggr(cls_tokens_local),
            #     key=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :]),
            #     value=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :])
            # )[0]  # 聚合局部特征
            #
            # output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()  # 通过边界框头获得输出坐标并应用sigmoid激活
            #
            # loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)  # 计算边界框损失和广义IoU损失
            
            # ##================= TMG ========================##
            # token_label = text.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
            # token_label[token_label==0] = -100 # -100 index = padding token
            # token_label[token_label==1] = 0
            #
            # for batch_idx in range(len(fake_text_pos)):
            #     fake_pos_sample = fake_text_pos[batch_idx]
            #     if fake_pos_sample:
            #         for pos in fake_pos_sample:
            #             token_label[batch_idx, pos] = 1
            #
            # input_ids = text.input_ids.clone()
            #
            # if self.args.token_momentum:
            #     with torch.no_grad():
            #         logits_m = self.text_encoder_m(input_ids,
            #                                     attention_mask = text.attention_mask,
            #                                     encoder_hidden_states = image_embeds_m,
            #                                     encoder_attention_mask = image_atts,
            #                                     return_dict = True,
            #                                     return_logits = True,
            #                                     )
            #     token_cls_output = self.text_encoder(input_ids,
            #                                 attention_mask = text.attention_mask,
            #                                 encoder_hidden_states = image_embeds,
            #                                 encoder_attention_mask = image_atts,
            #                                 return_dict = True,
            #                                 labels = token_label,
            #                                 soft_labels = F.softmax(logits_m.view(-1, 2),dim=-1),
            #                                 alpha = alpha
            #                                 )
            # else:
            #     token_cls_output  = self.text_encoder(input_ids,
            #                                 attention_mask = text.attention_mask,
            #                                 encoder_hidden_states = image_embeds,
            #                                 encoder_attention_mask = image_atts,
            #                                 return_dict = True,
            #                                 labels = token_label,
            #                                 )
            #
            # loss_TMG = token_cls_output.loss

            return loss_MAC, loss_BIC
                    # ,loss_TMG, loss_bbox, loss_giou,loss_MLC

        else:
            image_embeds = self.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state

            # forward the positve image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )               
            # ##================= IMG ========================##
            # bs = image.size(0)
            # cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)
            #
            # text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            # local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token
            #
            # local_feat_it_cross_attn = image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds),
            #                                   key=self.norm_layer_it_cross_atten(text_embeds),
            #                                   value=self.norm_layer_it_cross_atten(text_embeds),
            #                                   key_padding_mask=local_feat_padding_mask_text)[0]
            #
            # local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local),
            #                                   key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]),
            #                                   value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            # # 输出坐标
            # # output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            ##================= BIC ========================## 
            logits_real_fake = self.itm_head(output_pos.last_hidden_state[:,0,:])
            # ##================= MLC ========================##
            # logits_multicls = self.cls_head(output_pos.last_hidden_state[:,0,:])
            # ##================= TMG ========================##
            # input_ids = text.input_ids.clone()
            # logits_tok = self.text_encoder(input_ids,
            #                             attention_mask = text.attention_mask,
            #                             encoder_hidden_states = image_embeds,
            #                             encoder_attention_mask = image_atts,
            #                             return_dict = True,
            #                             return_logits = True,
            #                             )
            return logits_real_fake
                    # , logits_tok) ,logits_multicls,,output_coord


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()  # 禁用梯度计算，以减少内存使用和提高计算效率
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # 直接使用当前批次的图像特征和文本特征
        image_feats = image_feat  # 当前批次的图像特征
        text_feats = text_feat  # 当前批次的文本特征

        batch_size = image_feats.shape[0]  # 获取当前批次的大小

        ptr = int(self.queue_ptr)  # 获取当前队列的指针位置
        assert self.queue_size % batch_size == 0  # 确保队列大小可以被批次大小整除，便于管理

        # 如果队列已满，则清空指针
        if ptr + batch_size > self.queue_size:
            ptr = 0  # 重置指针

        # 在队列中替换指针位置的特征（出队和入队操作）
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T  # 将图像特征存入队列
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T  # 将文本特征存入队列
        ptr = (ptr + batch_size) % self.queue_size  # 更新指针，确保其在队列大小范围内循环

        self.queue_ptr[0] = ptr  # 将新的指针位置保存到队列指针中


# 目的是在分布式训练环境中收集所有进程的张量
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

