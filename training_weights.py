# I used Colab for training
import os
import glob
import math
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from einops import rearrange
from kornia.filters import GaussianBlur2d
from mobileone import MobileOne, mobileone
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm
import torch._dynamo

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_v, n_head, split, dropout, d_qk, compute_v, use_argmax=False):
        super().__init__()
        self.d_v = d_v
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_ks = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_v, bias=False)
        self.attention = None
        self.d_k = d_qk
        self.use_argmax = use_argmax

    def forward(self, q, k, v, qpos, kpos, qk_mask=None, k_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = v
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = torch.matmul(q / self.d_k**0.5, k.transpose(2, 3))
        if qk_mask is not None:
            attn += qk_mask
        attn = F.softmax(attn, dim=-1)
        if self.use_argmax:
            idx =  torch.argmax(attn, dim=1, keepdims=True)
            attn = torch.zeros_like(attn).scatter_(1, idx, 1.)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        return output, attn

class PatchInpainting(nn.Module):
    def __init__(self, *, kernel_size, nheads, stem_out_stride=1, stem_out_channels=3, cross_attention=False, mask_query_with_segmentation_mask=False, merge_mode='sum', use_kpos=True, image_size=512, embed_dim=512, use_qpos=True, dropout=0.1, attention_type='ane_transformers.reference.multihead_attention.SelfAttention', compute_v=0.1, feature_i=3, feature_dim=128, concat_features=True, attention_masking=True, final_conv=False, mask_inpainting=True, use_argmax=False, model):
        self.cross_attention = cross_attention
        self.kernel_size = kernel_size
        self.mask_query_with_segmentation_mask = mask_query_with_segmentation_mask
        self.nheads = nheads
        self.use_kpos = use_kpos
        self.use_qpos = use_qpos
        self.feature_i = feature_i
        self.feature_dim = feature_dim
        self.concat_features = concat_features
        self.attention_masking = attention_masking
        self.window_size= image_size // kernel_size
        self.final_conv = final_conv
        self.mask_inpainting = mask_inpainting
        self.use_argmax = use_argmax
        super().__init__()
        self.final_gaussian_blur = GaussianBlur2d((7,7),sigma=(2.01, 2.01),separable=False)
        self.pooling_layer = nn.MaxPool2d(kernel_size, stride=kernel_size)
        self.multihead_attention = MultiHeadAttention(embed_dim=stem_out_channels*kernel_size*kernel_size + self.feature_dim if self.concat_features else stem_out_channels*kernel_size*kernel_size, d_v=stem_out_channels*kernel_size*kernel_size, n_head=self.nheads, split=True, dropout=dropout, d_qk=embed_dim, compute_v=compute_v, use_argmax=self.use_argmax)
        self.stem_out_channels = stem_out_channels
        self.stem_out_stride = stem_out_stride
        self.register_buffer('qk_mask', 1e4 * torch.eye(int((image_size / stem_out_stride/self.kernel_size)**2)).unsqueeze(0).unsqueeze(0))
        if not mask_query_with_segmentation_mask:
            self.mask_query = torch.nn.Parameter(torch.zeros(1, int((image_size/stem_out_stride/self.kernel_size)**2), 1, 1).float())
        self.encoder_decoder = model
        self.image_size = image_size
        self.positionalencoding = torch.nn.Parameter(torch.zeros(1, self.kernel_size**2*stem_out_channels + self.feature_dim, int((image_size/stem_out_stride/self.kernel_size)**2))) if use_kpos or use_qpos else None
        self.final_conv = torch.nn.Sequential(nn.Conv2d(stem_out_channels*kernel_size*kernel_size, stem_out_channels*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, padding_mode='reflect'), torch.nn.Sigmoid()) if self.final_conv else None
        self.pixel_shuffle = nn.PixelShuffle(self.kernel_size)
        if merge_mode == 'all':
            self.merge_func = self.merge_all_patches_sum
        self.register_buffer(name="unfolding_weights", tensor=self._compute_unfolding_weights(self.kernel_size, self.stem_out_channels), persistent=False)
        self.register_buffer(name="unfolding_weights_image", tensor=self._compute_unfolding_weights(self.kernel_size, 3), persistent=False)
        self.register_buffer(name="unfolding_weights_mask", tensor=self._compute_unfolding_weights(self.kernel_size, 1), persistent=False)

    def _compute_unfolding_weights(self, kernel_size, channels) -> torch.Tensor:
        weights = torch.eye(kernel_size * kernel_size, dtype=torch.float)
        weights = weights.reshape((kernel_size * kernel_size, 1, kernel_size, kernel_size))
        weights = weights.repeat(channels, 1, 1, 1)
        return weights

    def unfolding_coreml(self, feature_map: torch.Tensor, weights, kernel_size: int):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.conv2d(feature_map, weights, bias=None, stride=(kernel_size, kernel_size), padding=0, dilation=1, groups=in_channels)
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: torch.Tensor, output_size, kernel_size: int, use_final_conv: bool) -> torch.Tensor:
        if use_final_conv and self.final_conv:
            patches = rearrange(patches, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
            patches = self.final_conv(patches)
            patches = rearrange(patches, 'b c p1 p2 -> b (p1 p2) c')
        final_image = rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=output_size[0]//kernel_size, w=output_size[1]//kernel_size, p1=kernel_size, p2=kernel_size)
        return final_image

    def forward(self, image, mask):
        image_coarse_inpainting, features = self.encoder_decoder(image)
        if self.mask_inpainting:
            image = image_coarse_inpainting*mask + image * (1 - mask)
        else:
            image = image_coarse_inpainting
        image_to_return = image_coarse_inpainting
        image_blurred = self.final_gaussian_blur(image)
        image_as_patches_blurred, _ = self.unfolding_coreml(image_blurred, self.unfolding_weights, self.kernel_size)
        image_as_patches, sizes = self.unfolding_coreml(image, self.unfolding_weights, self.kernel_size)
        image_as_patches = image_as_patches - image_as_patches_blurred
        pos = self.positionalencoding.repeat(image_as_patches.size(0), 1, 1).unsqueeze(2) if self.use_qpos else None
        mask_same_res_as_features_pooled, _ = self.unfolding_coreml(mask, self.unfolding_weights_mask, self.kernel_size)
        mask_same_res_as_features_pooled = mask_same_res_as_features_pooled[:, 0:1, :, :]
        mask_same_res_as_features_pooled = mask_same_res_as_features_pooled.flatten(start_dim=2).unsqueeze(-1)
        if self.concat_features:
            features_to_concat = features[self.feature_i]
            features_to_concat = F.interpolate(features_to_concat, size=image_as_patches.shape[-2:], mode='bilinear', align_corners=False)
            input_attn = torch.cat([image_as_patches, features_to_concat],dim=1)
            input_attn = input_attn.flatten(start_dim=2).transpose(1, 2)
        else:
            input_attn = image_as_patches.flatten(start_dim=2).transpose(1, 2)
        image_as_patches = image_as_patches.flatten(start_dim=2).transpose(1, 2)
        qk_mask = -1e4*self.qk_mask.repeat(image_as_patches.size(0), 1, 1, 1) + 2e4*((1 - mask_same_res_as_features_pooled)*self.qk_mask) if self.attention_masking else None
        k_mask  = -1e4*mask_same_res_as_features_pooled if self.attention_masking else None
        out, atten_weights = self.multihead_attention(input_attn, input_attn, image_as_patches, qpos=pos, kpos=pos, qk_mask=qk_mask, k_mask=k_mask)
        out = out - image_as_patches_blurred.flatten(start_dim=2).transpose(1, 2)
        mask = mask_same_res_as_features_pooled.squeeze(1).squeeze(-1).unsqueeze(-1)
        out = out * mask + image_as_patches * (1 - mask)
        out = self.folding_coreml(out, sizes, self.kernel_size, use_final_conv=True)
        return out, atten_weights, image_to_return

    def merge_all_patches_sum(self, patch_scores, sequence_of_patches):
        return torch.einsum('bkhq,bchk->bchq', patch_scores, sequence_of_patches.unsqueeze(2)).squeeze(2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2, :] = torch.sin(position * div_term).T
        pe[0, 1::2, :] = torch.cos(position * div_term).T
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe[:, :, :x.size(-1)]
        return self.dropout(x)

class MobileOneCoarse(nn.Module):
    def __init__(self, variant='s4', **kwargs):
        super().__init__()
        self.model = mobileone(variant=variant, **kwargs)
        dummy_input = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            x0 = self.model.stage0(dummy_input)
            x1 = self.model.stage1(x0)
            x2 = self.model.stage2(x1)
            x3 = self.model.stage3(x2)
            x4 = self.model.stage4(x3)
        self.channels = [x0.shape[1], x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]]
        c0, c1, c2, c3, c4 = self.channels
        self.d4 = nn.ConvTranspose2d(c4, 1792, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(1792 + c3, 896, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(896 + c2, 384, kernel_size=4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(384 + c1, 64, kernel_size=4, stride=2, padding=1)
        self.d0 = nn.ConvTranspose2d(64 + c0, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = []
        x0 = self.model.stage0(x)
        features.append(x0)
        x1 = self.model.stage1(x0)
        features.append(x1)
        x2 = self.model.stage2(x1)
        features.append(x2)
        x3 = self.model.stage3(x2)
        features.append(x3)
        x4 = self.model.stage4(x3)
        features.append(x4)
        out = self.relu(self.d4(x4))
        out = torch.cat([out, x3], dim=1)
        out = self.relu(self.d3(out))
        out = torch.cat([out, x2], dim=1)
        out = self.relu(self.d2(out))
        out = torch.cat([out, x1], dim=1)
        out = self.relu(self.d1(out))
        out = torch.cat([out, x0], dim=1)
        out = self.sigmoid(self.d0(out))
        return out, features

class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module: "PatchInpainting"):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

    def forward(self, x_hr, x_lr_inpainted, attn_map):
        hr_h, hr_w = x_hr.shape[-2:]
        lr_h, lr_w = x_lr_inpainted.shape[-2:]
        x_hr_base = F.interpolate(x_lr_inpainted, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        hr_patch_size = self.patch_inpainting.kernel_size * (hr_h // lr_h)
        unfolding_weights_hr = self.patch_inpainting._compute_unfolding_weights(kernel_size=hr_patch_size, channels=x_hr.shape[1]).to(x_hr.device)
        hr_patches, _ = self.patch_inpainting.unfolding_coreml(x_hr, unfolding_weights_hr, hr_patch_size)
        hr_blurred = self.patch_inpainting.final_gaussian_blur(x_hr)
        hr_patches_blurred, _ = self.patch_inpainting.unfolding_coreml(hr_blurred, unfolding_weights_hr, hr_patch_size)
        hr_hf_patches = hr_patches - hr_patches_blurred
        hr_hf_patches = hr_hf_patches.flatten(start_dim=2).transpose(1, 2)
        reconstructed_hr_hf_patches = torch.matmul(attn_map.squeeze(1), hr_hf_patches)
        reconstructed_hr_hf_image = self.patch_inpainting.folding_coreml(reconstructed_hr_hf_patches, (hr_h, hr_w), kernel_size=hr_patch_size, use_final_conv=False)
        final_hr_image = x_hr_base + reconstructed_hr_hf_image
        return final_hr_image

class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coarse_model = MobileOneCoarse(**config['coarse_model']['parameters'])
        feature_i = config['generator']['params'].get('feature_i', 2)
        correct_feature_dim = self.coarse_model.channels[feature_i]
        config['generator']['params']['feature_dim'] = correct_feature_dim
        self.generator = PatchInpainting(**config['generator']['params'], model=self.coarse_model)

    def forward(self, image, mask):
        return self.generator(image, mask)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = torch.nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]
        ]).eval()
        for param in self.blocks.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0.0
        x = (x - 0.5) / 0.5
        y = (y - 0.5) / 0.5
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class HighResInpaintingDataset(Dataset):
    def __init__(self, image_dir, size=1024):
        self.image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Error: Directory is empty or does not exist -> {image_dir}")
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def random_free_form_mask(self):
        mask = np.zeros((self.size, self.size), np.float32)
        num_strokes = np.random.randint(3, 10)
        for _ in range(num_strokes):
            start_x, start_y = np.random.randint(0, self.size, 2)
            end_x, end_y = np.random.randint(0, self.size, 2)
            thickness = np.random.randint(20, 100)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, thickness)
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        mask = self.random_free_form_mask()
        return img, mask

def train_weights():
    TOTAL_EPOCHS = 600
    accumulation_steps = 4
    START_EPOCH = 0
    checkpoint_path = "/content/drive/MyDrive/rethined_checkpoint.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HighResInpaintingDataset("/content/rethined/datasets/SuperCAF/images")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)

    config = {
        'coarse_model': {'class': 'MobileOneCoarse', 'parameters': {'variant': 's4'}},
        'generator': {
            'generator_class': 'PatchInpainting',
            'params': {
                'kernel_size': 8, 'nheads': 1, 'stem_out_stride': 1, 'stem_out_channels': 3,
                'merge_mode': 'all', 'image_size': 512, 'embed_dim': 576, 'use_qpos': None,
                'use_kpos': None, 'dropout': 0.1, 'feature_i': 2, 'concat_features': True,
                'final_conv': True, 'feature_dim': 896, 'attention_type': 'MultiHeadAttention',
                'compute_v': False, 'use_argmax': False
            }
        }
    }

    model = InpaintingModel(config).to(device)
    attention_upscaler = AttentionUpscaling(model.generator).to(device)

    is_resuming = False

    if os.path.exists(checkpoint_path):
        print("Found complete checkpoint. Restoring automatically...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

        START_EPOCH = checkpoint['epoch'] + 1
        is_resuming = True
        print(f"Resuming training from Epoch {START_EPOCH}")

    else:
        print("No previous data found. Starting training from scratch (Epoch 0).")

    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
    current_lr = 1e-4 

    torch._dynamo.config.suppress_errors = True
    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = AdamW(model.parameters(), lr=current_lr, weight_decay=1e-5)

    if is_resuming and os.path.exists(checkpoint_path):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully restored 100% of AdamW optimizer state.")
        except Exception as e:
            print(f"Warning: Could not load optimizer ({e}). Reinitialising.")

    l1_loss_fn = torch.nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    epochs_left = TOTAL_EPOCHS - START_EPOCH
    if epochs_left <= 0:
        print("The model has completed the required number of epochs.")
        return

    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=epochs_left * (len(dataloader) // accumulation_steps),
                                  eta_min=1e-6)

    model.train()

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")

        optimizer.zero_grad(set_to_none=True)

        for i, (hr_img, hr_mask) in progress_bar:
            hr_img, hr_mask = hr_img.to(device, non_blocking=True), hr_mask.to(device, non_blocking=True)

            lr_img = F.interpolate(hr_img, size=512, mode='bicubic', antialias=True)
            lr_mask = F.interpolate(hr_mask, size=512)
            masked_lr_img = lr_img * (1 - lr_mask)

            with torch.amp.autocast('cuda'):
                lr_out, attn_scores, coarse_out = model(masked_lr_img, lr_mask)
                hr_out = attention_upscaler(hr_img, lr_out, attn_scores)

                loss_l1_coarse = l1_loss_fn(coarse_out, lr_img)
                loss_l1_lr = l1_loss_fn(lr_out, lr_img)
                loss_l1_hr = l1_loss_fn(hr_out, hr_img)

                hr_out_for_vgg = F.interpolate(hr_out, size=1024, mode='bilinear')
                hr_img_for_vgg = F.interpolate(hr_img, size=1024, mode='bilinear')

                loss_perc_lr = perceptual_loss_fn(lr_out, lr_img)
                loss_perc_hr = perceptual_loss_fn(hr_out_for_vgg, hr_img_for_vgg)

                total_loss = (loss_l1_coarse + loss_l1_lr + loss_l1_hr + 0.1 * (loss_perc_lr + loss_perc_hr)) / accumulation_steps

            scaler.scale(total_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            display_loss = total_loss.item() * accumulation_steps
            epoch_loss += display_loss

            current_display_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'loss': f"{display_loss:.4f}", 'lr': f"{current_display_lr:.6f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} completed - Average Loss: {avg_loss:.4f}")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Successfully saved checkpoint (Epoch {epoch+1}) to Google Drive.")

if __name__ == "__main__":
    train_weights()
