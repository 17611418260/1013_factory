from torchvision.models.resnet import ResNet, Bottleneck
import pdb
import os
import gc
import cv2
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tifffile as tiff
import matplotlib.pyplot as plt
from fastai.vision.all import *
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


bs = 64
sz = 256    # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.225  # threshold for positive predictions  
DATA = './test_images/'
# MODELS = [f'../input/hubmap-models/hubmap_models/run_0/model_{i}.pth' for i in range(4)] + [f'../input/hubmap-models/hubmap_models/run_1/model_{i}.pth' for i in range(4)] + [f'../input/training-fastai-baseline/model_{i}.pth' for i in range(4)]
MODELS = [f'./model/model_{i}.pth' for i in range(4)]
df_sample = pd.read_csv('./sample_submission.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# * Data
# https://www.kaggle.com/datasets/thedevastator/hubmap-2022-256x256
mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

s_th = 40  # saturation blancking threshold
p_th = 1000*(sz//256)**2  # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.data = rasterio.open(os.path.join(DATA, idx+'.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce*sz
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0)//self.sz
        self.n1max = (self.shape[1] + self.pad1)//self.sz

    def __len__(self):
        return self.n0max*self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx//self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0+self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0+self.sz, self.shape[1])
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00-x0):(p01-x0), (p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1, 2, 3],
                                                                                   window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[(p00-x0):(p01-x0), (p10-y0):(p11-y0), i] =\
                    layer.read(1, window=Window.from_slices(
                        (p00, p01), (p10, p11)))

        if self.reduce != 1:
            img = cv2.resize(img, (self.sz//reduce, self.sz//reduce),
                             interpolation=cv2.INTER_AREA)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            # images with -1 will be skipped
            return img2tensor((img/255.0 - mean)/std), -1
        else:
            return img2tensor((img/255.0 - mean)/std), idx

# iterator like wrapper that returns predicted masks


class Model_pred:
    def __init__(self, models, dl, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                if ((y >= 0).sum() > 0):  # exclude empty images
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        # x,y,xy flips as TTA
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= (1+len(flips))
                    py /= len(self.models)

                    py = F.upsample(py, scale_factor=reduce, mode="bilinear")
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], y[i]
                        count += 1

    def __len__(self):
        return len(self.dl.dataset)

# * Model


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2**(len(self.convs)-i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4)
             for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1,
                                                   stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        # encoder
        m = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
        # m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
        #                   'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                                  m.layer1)  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        # aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[
                         stride*1, stride*2, stride*3, stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16]*4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(
            32+16*4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x


models = []
for path in MODELS:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model = UneXt50()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    models.append(model)

del state_dict

# mask_patch拼接
def reconstruct_image(patch_prefix, output_filename):
    patches = []

    # 读取所有patch
    for i in range(9):
        patch_filename = f"E:/hubmap_organ_segmentation/masks/{patch_prefix}{i}.png"
        patch = Image.open(patch_filename)
        patches.append(patch)

    # 获取每个patch的大小
    patch_width, patch_height = patches[0].size

    # 计算原始图片的大小
    original_width = 3 * patch_width
    original_height = 3 * patch_height

    # 创建一个新的Image对象，用于存储还原后的图片
    reconstructed_image = Image.new(
        "1", (original_width, original_height))  # 生成二值图

    # 将每个patch放回原始图片的相应位置
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            x_offset = j * patch_width
            y_offset = i * patch_height
            # 将PngImageFile对象转换为NumPy数组，并乘以255
            patch_array = np.array(patches[index]) * 255
            # 创建Image对象，并将NumPy数组转换回图像
            patch_image = Image.fromarray(patch_array.astype(np.uint8))
            reconstructed_image.paste(patch_image, (x_offset, y_offset))

    return reconstructed_image
    # # 保存还原后的图片
    # reconstructed_image.save(output_filename)


# * Predict
if __name__ == '__main__':
    names, preds = [], []
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        tiff_path = f"E:/hubmap_organ_segmentation/test_images/{row['id']}.tiff"
        tiff_gt = tiff.imread(tiff_path)
        plt.subplot(1, 3, 1)
        plt.imshow(tiff_gt)
        plt.title('tiff_image')

        patch_prefix = f"{row['id']}_000"
        output_filename = f"concat_{row['id']}.png"
        concat_mask = reconstruct_image(patch_prefix, output_filename)
        concat_mask = concat_mask.resize((3000, 3000), Image.NEAREST)
        plt.subplot(1, 3, 2)
        plt.imshow(concat_mask)
        plt.title('gt_mask')

        idx = str(row['id'])
        ds = HuBMAPDataset(idx)
        # rasterio cannot be used with multiple workers
        dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
        mp = Model_pred(models, dl)
        # generate masks
        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in iter(mp):
            mask[i.item()] = p.squeeze(-1) > TH

        # reshape tiled masks into a single mask and crop padding
        mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz).\
            permute(0, 2, 1, 3).reshape(ds.n0max*ds.sz, ds.n1max*ds.sz)
        mask = mask[ds.pad0//2:-(ds.pad0-ds.pad0//2) if ds.pad0 > 0 else ds.n0max*ds.sz,
                    ds.pad1//2:-(ds.pad1-ds.pad1//2) if ds.pad1 > 0 else ds.n1max*ds.sz]
        # pdb.set_trace()
        plt.subplot(1, 3, 3)
        plt.imshow(mask)
        plt.title('pred_mask')
        plt.tight_layout()
        plt.show()
