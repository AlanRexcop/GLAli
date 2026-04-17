import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from utils.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip_clear as clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from .zsclip_contra import CUSTOM_TEMPLATES
import os
import json
from copy import deepcopy
from utils.bonder import CrossAttnBlock
from utils.loss import SupConLoss

from utils.data_manager import build_data_loader
from dassl.data.transforms import build_transform

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def get_dense_logits2(image_features, local_image_features, all_text_features, mean_text_features, topk=50):
    base_logits = image_features @ mean_text_features.T   
    image_features = image_features.unsqueeze(1)  
    all_image_features = local_image_features
    w = torch.einsum('bmd,bnd->bmn', image_features, all_image_features) 

    mean_text_features = mean_text_features.unsqueeze(0) 
    _,n_cls,d = mean_text_features.shape
    all_text_features = all_text_features.reshape(-1, n_cls, d)
    v = torch.einsum('mcd,ncd->mnc', mean_text_features, all_text_features)  
    v = F.softmax(v, dim=1)
    sim = torch.einsum('bmd,ncd->bcmn', all_image_features, all_text_features)  
    sim, idx = sim.topk(dim=2, k=topk)    
    idx = idx[:, 0, :, 0].unsqueeze(1)
    w = torch.gather(w, dim=2, index=idx)
    w = F.softmax(w, dim=-1)
    weight = torch.einsum('bdm,dnc->bcmn', w,v) 
    mat = sim * weight
    
    bias_logits = torch.sum(mat, dim=(-2,-1))
    logits = base_logits + bias_logits
    return logits


def get_supc_loss(g_img_feats, id_loc_feats, ood_loc_feats, text_stu, text_tea, label, n_class=99, topk=50):
    bs, k, d = id_loc_feats.shape
    _, n_disc, _ = text_tea.shape
    id_ex_label = label.unsqueeze(1).repeat(1, k)
    ood_ex_label = torch.full((bs,), n_class).cuda()
    text_ex_label = label.unsqueeze(1).repeat(1, n_disc)

    features = torch.cat([id_loc_feats, ood_loc_feats], dim=0)
    res_label = torch.cat([label, ood_ex_label], dim=0)

    loss = SupConLoss(temperature=0.5, base_temperature=0.5)(features=features, labels=res_label)
    return loss


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, cache_keys=None, cache_values=None):
        super().__init__()
        
        self.device = torch.device("cuda")
        clip_model.to(self.device)
        self.image_encoder = clip_model.visual
        self.zs_img_encoder = deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_features_store =[]
        self.cfg = cfg

        description_file = os.path.join('./description', f'{cfg.DATASET.NAME}.json')
        print(f'Using description file: {description_file}')
        llm_descriptions = json.load(open(description_file))
        text_features =[]
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        all_prompt =[]
        print(classnames)
        for classname in classnames:
            prompts =[]
            prompt = template.format(classname.replace("_", " "))
            prompts.append(prompt)

            for i in range(50):
                prompt_desc = prompt + ' ' + llm_descriptions[classname.replace("_", " ")][i]
                prompts.append(prompt_desc)
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            all_prompt.append(prompts)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    text_features.append(clip_model.encode_text(prompts)) 
                    
        self.all_prompt = torch.cat(all_prompt)

        text_features = torch.cat(text_features) 
        _, d = text_features.shape
        self.ndisc = 51
        text_features = text_features.view(self.ndisc, -1, d)
        
        # SAFE NORMALIZATION
        self.all_text_features_tea = F.normalize(text_features, p=2, dim=-1)
        text_features_mean = text_features.mean(dim=0)
        self.text_features_tea = F.normalize(text_features_mean, p=2, dim=-1)
        self.text_prototypes = self.all_text_features_tea   

        if cfg.is_bonder:
            self.bonder = CrossAttnBlock(512)
            self.bonder.to(self.dtype)

        # ---------------- TIP-ADAPTER-F MEMORY CACHE ----------------
        self.tip_adapter = None
        if cache_keys is not None:
            print("Initializing Exact Tip-Adapter-F Cache Parameters...")
            
            self.tip_beta = 5.5   
            self.tip_adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False).to(self.dtype).cuda()
            self.tip_adapter.weight = nn.Parameter(cache_keys) 
            
            # Weighted Average Parameter
            # CHANGE THIS LINE (Initialize at -2.0 so sigmoid(-2.0) = ~0.12)
            # Starts by trusting original CLIP, slowly blends in the Cache to prevent shock
            self.tip_adapter.register_parameter("cache_scale", nn.Parameter(torch.tensor(-2.0, dtype=self.model.dtype).cuda()))
            self.register_buffer("cache_values", cache_values.to(self.dtype).cuda())

    def forward(self, image, mask=None, labels = None):
        with torch.no_grad():
            image_features_tea, local_image_features_tea, _ = self.zs_img_encoder(image.to(self.dtype))
            # SAFE NORMALIZATION
            image_features_tea = F.normalize(image_features_tea, p=2, dim=-1)
        
        image_features, local_image_features, _  = self.image_encoder(image.to(self.dtype))
        
        # SAFE NORMALIZATION
        image_features = F.normalize(image_features, p=2, dim=-1)
        local_image_features = F.normalize(local_image_features, p=2, dim=-1)

        text_prototypes = self.text_prototypes.detach()
        n_disc, c, d = text_prototypes.shape
        id_loc_feats = None
        ood_loc_feats = None
        l2p = None
        l2p_tea = None
        
        if labels is not None and self.cfg.is_bonder:
            bs = labels.shape[0]
            l2p = text_prototypes[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
            l2p_tea = self.all_text_features_tea[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
            l2p = torch.transpose(l2p, 0, 1)
            l2p_tea = torch.transpose(l2p_tea, 0, 1)

            sim = local_image_features @ (l2p.mean(dim=1, keepdim=True).transpose(1,2))
            sim = sim.squeeze(-1)
            _, idx = torch.topk(input=sim, k=self.cfg.topk)
            _, idx_ood = torch.topk(input=sim, k=self.cfg.topk, largest=False)

            l2p_loc = l2p[:, 1:, :]
            selected_loc_img_feats = torch.gather(local_image_features, 1, idx.unsqueeze(-1).expand(-1, -1, d))

            id_loc_feats = selected_loc_img_feats
            ood_loc_feats = torch.gather(local_image_features, 1, idx_ood.unsqueeze(-1).expand(-1, -1, d))
            
            text_bias = self.bonder(l2p_loc, selected_loc_img_feats.detach())
            # SAFE NORMALIZATION
            text_bias = F.normalize(text_bias, p=2, dim=-1)
            alpha = self.cfg.lambda_value
            updated_proto = self.text_prototypes
            
            contra_labels = torch.arange(c).view(-1,1).cuda()
            mask = torch.eq(labels.unsqueeze(1), contra_labels.T).to(self.dtype).cuda()
            update_features = torch.matmul(mask.view(bs, c).transpose(0,1).unsqueeze(0).repeat(n_disc-1,1,1), text_bias.transpose(1, 0))
            proto_mask = torch.zeros(c, dtype=torch.int).cuda()
            proto_mask[labels] = 1
            proto_mask = proto_mask.view(1, -1, 1).repeat(n_disc, 1, d)
            update_features = torch.cat([self.text_prototypes[0:1, :, :], update_features], dim=0)
            updated_proto = (1-proto_mask) * updated_proto + proto_mask * (alpha * updated_proto + (1-alpha) * update_features)

            # SAFE NORMALIZATION
            updated_proto_norm = F.normalize(updated_proto, p=2, dim=-1)
            updated_proto_mean = updated_proto_norm.mean(dim=0)
            updated_proto_mean_norm = F.normalize(updated_proto_mean, p=2, dim=-1)
        else:
            # SAFE NORMALIZATION
            updated_proto_norm = F.normalize(self.text_prototypes, p=2, dim=-1)
            updated_proto_mean = updated_proto_norm.mean(dim=0)
            updated_proto_mean_norm = F.normalize(updated_proto_mean, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * get_dense_logits2(image_features.detach(), local_image_features.detach(), updated_proto_norm, updated_proto_mean_norm, topk=self.cfg.topk)
        logits_local = logit_scale * get_dense_logits2(image_features, local_image_features, self.all_text_features_tea.detach(), self.text_features_tea.detach(), topk=self.cfg.topk)

        # ---------------- IMPROVED: LESION-ONLY TIP-ADAPTER ----------------
        if getattr(self, "tip_adapter", None) is not None:
            text_tea = self.text_features_tea.to(local_image_features.device).to(self.dtype)
            
            sim_to_all = torch.matmul(local_image_features, text_tea.T)
            max_sim_per_patch, _ = torch.max(sim_to_all, dim=-1)
            _, idx_lesion = torch.topk(max_sim_per_patch, k=self.cfg.topk, dim=1)
            
            lesion_patches = torch.gather(local_image_features, 1, idx_lesion.unsqueeze(-1).expand(-1, -1, d))
            lesion_query = F.normalize(lesion_patches.mean(dim=1), p=2, dim=-1)
            
            affinity = self.tip_adapter(lesion_query)
            
            cache_logits = torch.exp(-self.tip_beta * (1.0 - affinity)) @ self.cache_values.type_as(affinity)
            
            scale = torch.sigmoid(self.tip_adapter.cache_scale)
            
            logits = (logits * (1 - scale)) + (cache_logits * logit_scale * scale)
            logits_local = (logits_local * (1 - scale)) + (cache_logits * logit_scale * scale)
        # -------------------------------------------------------------------

        return logits, logits_local, image_features_tea, image_features, updated_proto_norm, id_loc_feats, ood_loc_feats, l2p, l2p_tea


@TRAINER_REGISTRY.register()
class LocProto(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LOCOOP.PREC in["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.lambda_value = cfg.lambda_value
        self.top_k = cfg.topk
        self.label =[]

        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.LOCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        self.model = CustomCLIP(cfg, classnames, clip_model, cache_keys=None, cache_values=None)
        self.model.to(self.device)

        # ---------------- CONSTRUCT LESION-ONLY TIP-ADAPTER CACHE ----------------
        tfm_test = build_transform(cfg, is_train=False)
        cache_loader = build_data_loader(
            cfg, sampler_type="SequentialSampler", data_source=self.dm.dataset.train_x,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, tfm=tfm_test, is_train=False
        )

        self.model.eval()
        cache_keys, cache_labels = [], []
        
        with torch.no_grad():
            text_tea = self.model.text_features_tea.to(self.device)
            for batch in tqdm(cache_loader, desc="Building Cache"):
                image, label = batch["img"].to(self.device), batch["label"].to(self.device)
                _, local_feat, _ = self.model.zs_img_encoder(image.type(self.model.dtype))
                # SAFE NORMALIZATION
                local_feat = F.normalize(local_feat, p=2, dim=-1)
                
                gt_text = text_tea[label].unsqueeze(1) 
                sim_to_gt = torch.bmm(local_feat, gt_text.transpose(1, 2)).squeeze(-1)
                _, idx_pos = torch.topk(sim_to_gt, k=self.top_k, dim=1)
                
                lesion_feats = torch.gather(local_feat, 1, idx_pos.unsqueeze(-1).expand(-1, -1, local_feat.shape[-1]))
                # SAFE NORMALIZATION
                lesion_feat_mean = F.normalize(lesion_feats.mean(dim=1), p=2, dim=-1)
                
                cache_keys.append(lesion_feat_mean.cpu())
                cache_labels.append(label.cpu())
                
        cache_keys = torch.cat(cache_keys, dim=0).to(dtype=self.model.dtype, device=self.device) 
        cache_labels = torch.cat(cache_labels, dim=0).to(self.device) 
        cache_values = F.one_hot(cache_labels, num_classes=len(classnames)).to(dtype=self.model.dtype, device=self.device) 

        # Inject into model
        self.model.tip_beta = 5.5   
        self.model.tip_adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False).to(self.model.dtype).cuda()
        self.model.tip_adapter.weight = nn.Parameter(cache_keys) 
        self.model.tip_adapter.register_parameter("cache_scale", nn.Parameter(torch.tensor(0.0, dtype=self.model.dtype).cuda()))
        self.model.register_buffer("cache_values", cache_values)

        # ---------------- GRADIENT CONFIGURATION ----------------
        for name, param in self.model.named_parameters():
            if 'image_encoder.transformer.resblocks.11.attn' in name or 'bonder' in name or 'tip_adapter' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        if "ViT" in cfg.MODEL.BACKBONE.NAME:
            
            # PREVENT JITTER: Force max LR to 1e-5 for dense ViT layers
            cfg_safe_attn = deepcopy(cfg.OPTIM)
            cfg_safe_attn.LR = min(cfg.OPTIM.LR, 1e-5) # Changed from 1e-4 to 1e-5
            
            self.optim = build_optimizer(self.model.image_encoder.transformer.resblocks[-1].attn, cfg_safe_attn)
            self.sched = build_lr_scheduler(self.optim, cfg_safe_attn)
            self.register_model("attn_learner", self.model.image_encoder.transformer.resblocks[-1].attn, self.optim, self.sched)
            
            if cfg.is_bonder:
                cfg_safe_bonder = deepcopy(cfg.OPTIM)
                cfg_safe_bonder.LR = min(cfg.OPTIM.LR, 1e-4)
                
                self.optim2 = build_optimizer(self.model.bonder, cfg_safe_bonder)
                self.sched2 = build_lr_scheduler(self.optim2, cfg_safe_bonder)
                self.register_model("bonder_learner", self.model.bonder, self.optim2, self.sched2)

            if hasattr(self.model, "tip_adapter") and self.model.tip_adapter is not None:
                tip_params = [
                    {"params": [self.model.tip_adapter.weight], "lr": 0.001},
                    {"params": [self.model.tip_adapter.cache_scale], "lr": 0.001} 
                ]
                # It's fine for the tip adapter adapter keys to have LR 0.001
                cfg_tip = deepcopy(cfg.OPTIM)
                cfg_tip.LR = 0.001
                self.optim_tip = build_optimizer(tip_params, cfg_tip)
                self.sched_tip = build_lr_scheduler(self.optim_tip, cfg_tip)
                self.register_model("tip_adapter_learner", self.model.tip_adapter, self.optim_tip, self.sched_tip)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None
        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LOCOOP.PREC

        if prec == "amp":
            with autocast():
                output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
                all_text_features_tea = self.model.all_text_features_tea.clone()
                
                # Smooth the loss landscape to prevent wild loss spikes
                loss_id = F.cross_entropy(output, label, label_smoothing=0.1)
                loss_id2 = F.cross_entropy(output_local, label, label_smoothing=0.1)
                loss_distil_img = F.l1_loss(img_feat_tea, img_feat_stu, reduction='mean') * 10
                loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
                loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
                loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            for name in self._optims:
                if self._optims[name] is not None: self._optims[name].zero_grad()
                
            self.scaler.scale(loss).backward()
            
            # SAFEGUARD: Clip gradients before stepping to prevent explosion
            for name in self._optims:
                if self._optims[name] is not None: 
                    self.scaler.unscale_(self._optims[name])
                    torch.nn.utils.clip_grad_norm_(self._models[name].parameters(), max_norm=1.0)
                    self.scaler.step(self._optims[name])
                    
            self.scaler.update()
        else:
            output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
            all_text_features_tea = self.model.all_text_features_tea.clone()
            
            # Smooth the loss landscape to prevent wild loss spikes
            loss_id = F.cross_entropy(output, label, label_smoothing=0.1)
            loss_id2 = F.cross_entropy(output_local, label, label_smoothing=0.1)
            loss_distil_img = F.l1_loss(img_feat_tea, img_feat_stu, reduction='mean') * 10
            loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
            loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
            loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            for name in self._optims:
                if self._optims[name] is not None: self._optims[name].zero_grad()
                
            loss.backward()
            
            # SAFEGUARD: Clip gradients
            for name in self._optims:
                if self._optims[name] is not None: 
                    torch.nn.utils.clip_grad_norm_(self._models[name].parameters(), max_norm=1.0)
                    self._optims[name].step()

        loss_summary = {
            "loss": loss.item(), "loss_id": loss_id.item(), "loss_distil_img": loss_distil_img.item(),
            "loss_distil_text": loss_distil_text.item(), "acc": compute_accuracy(output_local, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        self.model.text_prototypes = text_stu.detach()
        return loss_summary

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory: return
        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in names:
            model_path = osp.join(directory, name, model_file)
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            keys_to_delete =[k for k in state_dict.keys() if "token_prefix" in k or "token_suffix" in k]
            for k in keys_to_delete: del state_dict[k]
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        data_loader = self.val_loader if (split == "val" and self.val_loader is not None) else self.test_loader
        print(f"Evaluate on the *{split}* set")
        if self.cfg.is_bonder:
            self.model.text_prototypes = torch.load(osp.join(self.output_dir, 'proto.pth'))
        for batch in tqdm(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) >= 2:
                output = output[1] + 0.05 * output[0] if self.cfg.is_bonder else output[0]
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()
        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)
        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)
        self.set_model_mode("eval")
        mcm_score =[]
        for batch in tqdm(data_loader):
            images = batch[0].cuda()
            output, output_local, _, _, _, _, _, _, _ = self.model_inference(images)
            if self.cfg.is_bonder: output = output_local + 0.05 * output
            output /= 100.0
            smax_global = to_np(F.softmax(output/T, dim=-1))  
            mcm_score.append(-np.max(smax_global, axis=1))
        res = concat(mcm_score)[:len(data_loader.dataset)].copy()
        return res, res, res, res

    @torch.no_grad()
    def test_visualize(self, img_path, label_idx):
        self.set_model_mode("eval")
        from dassl.data.transforms import build_transform
        from PIL import Image
        tfm_test = build_transform(self.cfg, is_train=False)
        image = Image.open(img_path).convert("RGB")
        image_tensor = tfm_test(image).unsqueeze(0).to(self.device)
        _, local_features, _ = self.model.image_encoder(image_tensor.type(self.model.dtype))
        local_features = F.normalize(local_features, p=2, dim=-1)
        target_text = self.model.text_features_tea[label_idx] 
        patch_scores = (local_features[0] @ target_text).float()
        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() + 1e-8)
        heatmap = patch_scores.view(14, 14).cpu().numpy()
        return heatmap, image