import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.deep_supervision = False

        if getattr(cfg, 'use_weather_data', False):
            logger.info('Using gate for weather data')
            self.gate = nn.Conv2d(cfg.in_chans, cfg.in_chans, kernel_size=1, bias=False)
        else:
            logger.info('Not using gate for weather data')
            self.gate = nn.Identity()

        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'dinov3':
            logger.info('Using backbone: DINOv3')
            from .encoders.dual_dinov3 import DualDINOv3 as backbone
            use_adapter = getattr(cfg, 'dinov3_use_adapter', None)
            if use_adapter is None:
                use_adapter = (cfg.decoder == 'Mask2Former')
            self.backbone = backbone(
                model_name=getattr(cfg, 'dinov3_model_name', ''),
                repo_dir=getattr(cfg, 'dinov3_repo_dir', ''),
                pretrained_path=getattr(cfg, 'dinov3_pretrained', None),
                out_indices=getattr(cfg, 'dinov3_out_indices', None),
                norm_fuse=norm_layer,
                use_frm=getattr(cfg, 'dinov3_use_frm', True),
                use_adapter=use_adapter,
                    use_deform_attn=getattr(cfg, 'dinov3_use_deform_attn', False),
                    deform_n_heads=getattr(cfg, 'dinov3_deform_n_heads', 8),
                    deform_n_points=getattr(cfg, 'dinov3_deform_n_points', 4),
                freeze_backbone=getattr(cfg, 'dinov3_freeze_backbone', False),
            )
            self.channels = self.backbone.out_channels
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mamba':
            logger.info('Using backbone: MAMBA')
            from .encoders.dual_vmamba_sub_attn import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'sigma_tiny':
            logger.info('Using backbone: V-MAMBA')
            self.channels = [96, 192, 384, 768]
            from .encoders.dual_vmamba import vssm_tiny as backbone
            self.backbone = backbone()
        elif cfg.backbone == 'sigma_small':
            logger.info('Using backbone: V-MAMBA')
            self.channels = [96, 192, 384, 768]
            from .encoders.dual_vmamba import vssm_small as backbone
            # self.backbone = backbone(in_chans=cfg.in_chans)
            if getattr(cfg, 'use_weather_data', False):
                self.backbone = backbone(in_chans=cfg.in_chans)
            else:
                self.backbone = backbone()

        elif cfg.backbone == 'sigma_base':
            logger.info('Using backbone: V-MAMBA')
            self.channels = [128, 256, 512, 1024]
            from .encoders.dual_vmamba import vssm_base as backbone
            self.backbone = backbone()
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
            
        elif cfg.decoder == 'MambaDecoder':
            logger.info('Using Mamba Decoder')
            from .decoders.MambaDecoder import MambaDecoder
            self.deep_supervision = False
            self.decode_head = MambaDecoder(img_size=[cfg.image_height, cfg.image_width], in_channels=self.channels, num_classes=cfg.num_classes, embed_dim=self.channels[0], deep_supervision=self.deep_supervision)

        elif cfg.decoder == 'Mask2Former':
            logger.info('Using Mask2Former Decoder')
            from .decoders.Mask2FormerDecoder import Mask2FormerDecoder
            hidden_dim = getattr(cfg, 'mask2former_hidden_dim', getattr(cfg, 'decoder_embed_dim', 256))
            input_strides = getattr(cfg, 'mask2former_input_strides', (4, 8, 16, 32))
            mask2former_mode = getattr(cfg, 'mask2former_train_mode', 2)
            output_mode = 'raw' if mask2former_mode in (2, 4) else 'logits'
            self.decode_head = Mask2FormerDecoder(
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                hidden_dim=hidden_dim,
                input_strides=input_strides,
                output_mode=output_mode,
            )

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            if cfg.backbone != 'vmamba':
                logger.info('Loading pretrained model: {}'.format(pretrained))
                self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.gate, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        if not self.deep_supervision:
            orisize = rgb.shape
            x = self.backbone(rgb, modal_x)
            out = self.decode_head.forward(x)
            if isinstance(out, dict):
                return out
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out, aux_fm
            return out
        else:
            x = self.backbone(rgb, modal_x)
            x_last, x_output_0, x_output_1, x_output_2 = self.decode_head.forward(x)
            return x_last, x_output_0, x_output_1, x_output_2

    def forward(self, rgb, modal_x, label=None):

        
        rgb = self.gate(rgb)
        modal_x = self.gate(modal_x)


        if not self.deep_supervision:
            if self.aux_head:
                out, aux_fm = self.encode_decode(rgb, modal_x)
                # print("out shape: ", out.shape, " aux_fm shape: ", aux_fm.shape)
            else:
                out = self.encode_decode(rgb, modal_x)
                # print("out shape: ", out.shape)
            if isinstance(out, dict):
                if label is not None:
                    loss = self.criterion(out, label.long())
                    return loss
                if hasattr(self.decode_head, "semantic_logits_from_raw"):
                    out = self.decode_head.semantic_logits_from_raw(out, logits=True)
                    out = F.interpolate(out, size=rgb.shape[2:], mode='bilinear', align_corners=False)
                return out

            if label is not None:
                loss = self.criterion(out, label.long())
                if self.aux_head:
                    loss += self.aux_rate * self.criterion(aux_fm, label.long())
                # logger.info('loss: {}'.format(loss.item()))
                return loss
            return out
        else:
            x_last, x_output_0, x_output_1, x_output_2 = self.encode_decode(rgb, modal_x)
            if label is not None:
                loss = self.criterion(x_last, label.long())
                loss += self.criterion(x_output_0, label.long())
                loss += self.criterion(x_output_1, label.long())
                loss += self.criterion(x_output_2, label.long())
                return loss
            return x_last
        
    def flops(self, shape=(3, 256, 256)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy
        
        '''
        code from
        https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L4
        '''
        
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = (torch.randn((1, *shape), device=next(model.parameters()).device), torch.randn((1, *shape), device=next(model.parameters()).device))
        print(len(input))
        for i in input:
            print(i.shape)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"
    

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops
  
def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops
