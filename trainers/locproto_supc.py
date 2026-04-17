import os
import os.path as osp
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform

from utils.trainer import TrainerX
from utils.bonder import CrossAttnBlock
from utils.loss import SupConLoss
from utils.data_manager import build_data_loader

from clip_w_local import clip_clear as clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .zsclip_contra import CUSTOM_TEMPLATES

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
    """
    Computes global and local vision-language alignment.
    all_text_features expected shape: [n_cls, n_desc, d]
    """
    n_cls, n_desc, d = all_text_features.shape
    
    base_logits = image_features @ mean_text_features.T   
    img_f = image_features.unsqueeze(1)  
    
    w = torch.einsum('bmd,bnd->bmn', img_f, local_image_features) 

    mean_text = mean_text_features.unsqueeze(0) 
    
    # Reshape text to original function format: [n_desc, n_cls, d]
    all_text_f = all_text_features.transpose(0, 1)
    
    v = torch.einsum('mcd,ncd->mnc', mean_text, all_text_f)  
    v = F.softmax(v, dim=1)
    
    sim = torch.einsum('bmd,ncd->bcmn', local_image_features, all_text_f)  
    sim_topk, idx = sim.topk(dim=2, k=topk)    
    
    idx_w = idx[:, 0, :, 0].unsqueeze(1)
    w_topk = torch.gather(w, dim=2, index=idx_w)
    w_topk = F.softmax(w_topk, dim=-1)
    
    weight = torch.einsum('bdm,dnc->bcmn', w_topk, v) 
    mat = sim_topk * weight
    
    bias_logits = torch.sum(mat, dim=(-2,-1))
    logits = base_logits + bias_logits
    return logits


def get_supc_loss(global_feats, id_loc_feats, ood_loc_feats, label, n_class):
    """
    GLAli's Local Contrastive Loss constraint modified for standard SupConLoss.
    Pulls together global and top-k features of the same class, pushes apart other classes,
    and assigns all bottom-k irrelevant patches to a generic OOD label.
    """
    bs, k, d = id_loc_feats.shape
    
    # [bs, 1, d] and [bs, k, d] -> [bs, k+1, d]
    id_feats = torch.cat([global_feats.unsqueeze(1), id_loc_feats], dim=1) 
    ood_feats = ood_loc_feats 
    
    # Flatten everything as independent instance views
    features = torch.cat([id_feats.reshape(-1, d), ood_feats.reshape(-1, d)], dim=0).unsqueeze(1) # [N, 1, d]
    
    l_id = label.unsqueeze(1).repeat(1, k+1).reshape(-1)
    l_ood = torch.full((bs * k,), n_class, dtype=label.dtype, device=label.device)
    res_label = torch.cat([l_id, l_ood], dim=0)
    
    loss = SupConLoss(temperature=0.1)(features=features, labels=res_label)
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


class GL_DAC_Model(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.device = torch.device("cuda")
        clip_model.to(self.device)
        self.cfg = cfg
        
        # Core CLIP Encoders
        self.image_encoder = clip_model.visual
        self.zs_img_encoder = deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # ---------------- GL-DAC ARCHITECTURE MODULES ----------------
        d = self.image_encoder.output_dim
        
        # DAC Visual Adapter: A single linear layer bridging pre-trained & target domains
        self.visual_adapter = nn.Linear(d, d, bias=False).to(self.dtype).to(self.device)
        nn.init.eye_(self.visual_adapter.weight) 
        
        # GLAli Text Refinement Module (Cross Attention)
        self.bonder = CrossAttnBlock(d).to(self.dtype).to(self.device)

        # DAC Ensemble balancing parameter
        self.ensemble_scale = nn.Parameter(torch.tensor(-2.0, dtype=self.dtype).to(self.device))
        self.tip_beta = 5.5

        # ---------------- TEXT CACHE INITIALIZATION ----------------
        description_file = osp.join('./description', f'{cfg.DATASET.NAME}.json')
        print(f'Using LLM description file: {description_file}')
        llm_descriptions = json.load(open(description_file))
        
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        text_features = []
        
        for classname in classnames:
            prompts = []
            prompt = template.format(classname.replace("_", " "))
            prompts.append(prompt)
            for i in range(50):
                prompts.append(prompt + ' ' + llm_descriptions[classname.replace("_", " ")][i])
                
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad(), autocast():
                text_features.append(clip_model.encode_text(prompts))
                
        # [n_cls, n_desc, d]
        text_features = torch.cat(text_features).view(len(classnames), 51, d)
        
        # Base frozen text
        self.register_buffer("base_text_features", text_features.clone())
        
        # Learnable target text cache (DAC inter-modal fine-tuning)
        self.text_cache = nn.Parameter(text_features.clone()) 
        
        # ---------------- VISUAL CACHE BUFFERS ----------------
        self.register_buffer("raw_cache_local_feats", None)
        self.register_buffer("cache_labels", None)
        self.register_buffer("cache_values", None)

    def forward(self, image, labels=None):
        bs = image.shape[0]
        n_cls, n_desc, d = self.text_cache.shape
        
        # 1. Feature Extraction
        with torch.no_grad():
            _, raw_local, _ = self.zs_img_encoder(image.to(self.dtype))
            
        _, local_feat, _ = self.image_encoder(image.to(self.dtype))
        
        # 2. DAC Visual Adaptation
        adapted_local = self.visual_adapter(local_feat)
        adapted_local = F.normalize(adapted_local, p=2, dim=-1)
        adapted_global = F.normalize(adapted_local.mean(dim=1), p=2, dim=-1)

        # 3. Top-k and Bottom-k Extraction (GLAli)
        with torch.no_grad():
            base_text_mean = F.normalize(self.text_cache.mean(dim=1).detach(), p=2, dim=-1)
            sim_to_text = adapted_local @ base_text_mean.T # [bs, L, n_cls]
            
        if labels is not None:
            sim_target = sim_to_text[torch.arange(bs), :, labels] 
        else:
            sim_target, _ = sim_to_text.max(dim=-1) 
            
        _, topk_idx = torch.topk(sim_target, k=self.cfg.topk, dim=-1)
        _, botk_idx = torch.topk(sim_target, k=self.cfg.topk, dim=-1, largest=False)
        
        topk_feats = torch.gather(adapted_local, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
        botk_feats = torch.gather(adapted_local, 1, botk_idx.unsqueeze(-1).expand(-1, -1, d))

        # 4. GLAli Visually-Guided Text Refinement
        if self.training and labels is not None:
            # Refine corresponding ground-truth text prototypes using image patches
            l2p = self.text_cache[labels] # [bs, n_desc, d]
            text_bias = self.bonder(l2p, topk_feats.detach()) # [bs, n_desc, d]
            
            bias_full = torch.zeros_like(self.text_cache).unsqueeze(0).repeat(bs, 1, 1, 1) 
            for i in range(bs):
                bias_full[i, labels[i]] = text_bias[i]
            bias_avg = bias_full.mean(dim=0) # [n_cls, n_desc, d]
            refined_text = self.text_cache + self.cfg.lambda_value * bias_avg
        else:
            # Dynamic cross-attention refinement over all classes during inference
            tc_expand = self.text_cache.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs * n_cls, n_desc, d)
            tk_expand = topk_feats.unsqueeze(1).expand(-1, n_cls, -1, -1).reshape(bs * n_cls, self.cfg.topk, d)
            bias = self.bonder(tc_expand, tk_expand).reshape(bs, n_cls, n_desc, d)
            
            bias_avg = bias.mean(dim=0)
            refined_text = self.text_cache + self.cfg.lambda_value * bias_avg

        refined_text_norm = F.normalize(refined_text, p=2, dim=-1)
        refined_text_mean_norm = F.normalize(refined_text_norm.mean(dim=1), p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        
        # 5. Inter-Modal Logits (GLAli Adaptive Alignment)
        inter_logits = logit_scale * get_dense_logits2(
            adapted_global, adapted_local, 
            refined_text_norm, refined_text_mean_norm, 
            topk=self.cfg.topk
        )
        
        # 6. Intra-Modal Logits (DAC Hierarchical Visual Cache)
        intra_logits = torch.zeros_like(inter_logits)
        if self.raw_cache_local_feats is not None:
            # Adapt the memory cache dynamically
            cache_adapted = self.visual_adapter(self.raw_cache_local_feats.type_as(adapted_local))
            cache_adapted = F.normalize(cache_adapted, p=2, dim=-1)
            
            with torch.no_grad():
                cache_sim = cache_adapted @ base_text_mean.T
                cache_sim_target = cache_sim[torch.arange(cache_sim.shape[0]), :, self.cache_labels]
                _, cache_topk_idx = torch.topk(cache_sim_target, k=self.cfg.topk, dim=-1)
                
            cache_topk_feats = torch.gather(cache_adapted, 1, cache_topk_idx.unsqueeze(-1).expand(-1, -1, d))
            cache_keys = F.normalize(cache_topk_feats.mean(dim=1), p=2, dim=-1) # [N_cache, d]
            
            query_keys = F.normalize(topk_feats.mean(dim=1), p=2, dim=-1) # [bs, d]
            
            affinity = torch.exp(-self.tip_beta * (1.0 - query_keys @ cache_keys.T)) # [bs, N_cache]
            intra_logits = affinity @ self.cache_values.type_as(affinity)
            
        # 7. Final Ensembled Prediction
        scale = torch.sigmoid(self.ensemble_scale)
        logits = (inter_logits * (1 - scale)) + (intra_logits * logit_scale * scale)

        return logits, inter_logits, intra_logits, adapted_global, topk_feats, botk_feats, refined_text_norm


@TRAINER_REGISTRY.register()
class LocProto(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LOCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.LOCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        self.model = GL_DAC_Model(cfg, classnames, clip_model)
        self.model.to(self.device)

        # ---------------- BUILD DAC FEW-SHOT VISUAL CACHE ----------------
        tfm_test = build_transform(cfg, is_train=False)
        cache_loader = build_data_loader(
            cfg, sampler_type="SequentialSampler", data_source=self.dm.dataset.train_x,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, tfm=tfm_test, is_train=False
        )

        self.model.eval()
        raw_local_feats, cache_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(cache_loader, desc="Building Base Hierarchical Visual Cache"):
                image, label = batch["img"].to(self.device), batch["label"].to(self.device)
                _, local_feat, _ = self.model.zs_img_encoder(image.type(self.model.dtype))
                raw_local_feats.append(local_feat.cpu())
                cache_labels.append(label.cpu())
                
        self.model.raw_cache_local_feats = torch.cat(raw_local_feats, dim=0).to(self.device) 
        self.model.cache_labels = torch.cat(cache_labels, dim=0).to(self.device) 
        self.model.cache_values = F.one_hot(self.model.cache_labels, num_classes=len(classnames)).to(dtype=self.model.dtype, device=self.device) 

        # ---------------- GRADIENT CONFIGURATION ----------------
        for name, param in self.model.named_parameters():
            # Activate gradients for GL-DAC components
            if any(key in name for key in ['visual_adapter', 'bonder', 'text_cache', 'ensemble_scale']):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        cfg_safe = deepcopy(cfg.OPTIM)
        cfg_safe.LR = min(cfg.OPTIM.LR, 1e-4)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = build_optimizer(trainable_params, cfg_safe)
        self.sched = build_lr_scheduler(self.optim, cfg_safe)
        
        self.register_model("gldac_learner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LOCOOP.PREC

        with autocast(enabled=(prec == "amp")):
            output, _, _, adapted_global, topk_feats, botk_feats, _ = self.model(image, labels=label)
            
            # Extract Distillation Targets from Teacher CLIP
            with torch.no_grad():
                _, local_tea, _ = self.model.zs_img_encoder(image.type(self.model.dtype))
                local_tea = F.normalize(local_tea, p=2, dim=-1)
                global_tea = F.normalize(local_tea.mean(dim=1), p=2, dim=-1)
                
            loss_id = F.cross_entropy(output, label, label_smoothing=0.1)
            
            # Local Contrastive Loss pushes irrelevant patches to OOD mapping and groups ID features
            loss_supc = get_supc_loss(adapted_global, topk_feats, botk_feats, label, n_class=len(self.dm.dataset.classnames))
            
            # DAC Regularization: maintain core multi-modal alignment
            if isinstance(self.model, nn.DataParallel):
                loss_distil_text = F.l1_loss(self.model.module.text_cache, self.model.module.base_text_features) * 25
            else:
                loss_distil_text = F.l1_loss(self.model.text_cache, self.model.base_text_features) * 25
                
            loss_distil_img = F.l1_loss(adapted_global, global_tea) * 10
            
            loss = loss_id + (loss_supc * 0.5) + loss_distil_img + loss_distil_text

        for name in self._optims:
            if self._optims[name] is not None: self._optims[name].zero_grad()
            
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            for name in self._optims:
                if self._optims[name] is not None: 
                    self.scaler.unscale_(self._optims[name])
                    torch.nn.utils.clip_grad_norm_(self._models[name].parameters(), max_norm=1.0)
                    self.scaler.step(self._optims[name])
            self.scaler.update()
        else:
            loss.backward()
            for name in self._optims:
                if self._optims[name] is not None: 
                    torch.nn.utils.clip_grad_norm_(self._models[name].parameters(), max_norm=1.0)
                    self._optims[name].step()

        loss_summary = {
            "loss": loss.item(), "loss_id": loss_id.item(), "loss_supc": loss_supc.item(),
            "acc": compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

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
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        data_loader = self.val_loader if (split == "val" and self.val_loader is not None) else self.test_loader
        print(f"Evaluate on the *{split}* set")
        
        for batch in tqdm(data_loader):
            input, label = self.parse_batch_test(batch)
            output, _, _, _, _, _, _ = self.model(input)
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
        mcm_score = []
        
        for batch in tqdm(data_loader):
            images = batch[0].cuda()
            output, _, _, _, _, _, _ = self.model(images)
            
            output /= 100.0
            smax_global = to_np(F.softmax(output/T, dim=-1))  
            mcm_score.append(-np.max(smax_global, axis=1))
            
        res = concat(mcm_score)[:len(data_loader.dataset)].copy()
        return res, res, res, res

    @torch.no_grad()
    def test_visualize(self, img_path, label_idx):
        """
        Maintains visualization mechanics mapping locally-adapted features onto
        the fully enriched text representations.
        """
        self.set_model_mode("eval")
        tfm_test = build_transform(self.cfg, is_train=False)
        image = Image.open(img_path).convert("RGB")
        image_tensor = tfm_test(image).unsqueeze(0).to(self.device)
        
        mod = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        _, local_features, _ = mod.image_encoder(image_tensor.type(mod.dtype))
        adapted_local = mod.visual_adapter(local_features)
        adapted_local = F.normalize(adapted_local, p=2, dim=-1)
        
        target_text = mod.text_cache[label_idx].mean(dim=0)
        target_text = F.normalize(target_text, p=2, dim=-1)
        
        patch_scores = (adapted_local[0] @ target_text).float()
        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() + 1e-8)
        heatmap = patch_scores.view(14, 14).cpu().numpy()
        
        return heatmap, image