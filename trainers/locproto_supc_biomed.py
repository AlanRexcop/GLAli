import os
import os.path as osp
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

from dassl.engine import TRAINER_REGISTRY
from utils.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform

from utils.bonder import CrossAttnBlock
from utils.loss import SupConLoss
from utils.data_manager import build_data_loader
from trainers.zsclip_contra import entropy_select_topk2, CUSTOM_TEMPLATES

import open_clip
from clip_w_local.biomedclip import load_biomedclip

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
    ood_ex_label = torch.full((bs,), n_class).cuda()

    features = torch.cat([id_loc_feats, ood_loc_feats], dim=0)
    res_label = torch.cat([label, ood_ex_label], dim=0)

    loss = SupConLoss(temperature=0.5, base_temperature=0.5)(features=features, labels=res_label)
    return loss

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text.transformer
        self.proj = clip_model.text.proj

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled_out = outputs.last_hidden_state[:, 0, :]
        if self.proj is not None:
            if isinstance(self.proj, nn.Module):
                pooled_out = self.proj(pooled_out)
            else:
                pooled_out = pooled_out @ self.proj
        return pooled_out

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.LOCOOP.N_CTX
        ctx_init = cfg.TRAINER.LOCOOP.CTX_INIT
        
        word_embedding = clip_model.text.transformer.embeddings.word_embeddings
        ctx_dim = word_embedding.weight.shape[1]
        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_ids = tokenizer(ctx_init, return_tensors="pt").input_ids[0][1:-1]
            with torch.no_grad():
                embedding = word_embedding(prompt_ids.cuda())
            ctx_vectors = embedding
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        
        max_len = 256
        self.token_prefix = []
        self.token_suffix = []
        self.attention_mask = []
        
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        
        for name in classnames:
            name_ids = tokenizer(name, add_special_tokens=False).input_ids
            prefix = [cls_id]
            suffix = name_ids + [sep_id]
            pad_len = max_len - len(prefix) - n_ctx - len(suffix)
            suffix = suffix + [pad_id] * pad_len
            attn_mask = [1] * (len(prefix) + n_ctx + len(name_ids) + 1) + [0] * pad_len
            
            self.token_prefix.append(prefix)
            self.token_suffix.append(suffix)
            self.attention_mask.append(attn_mask)
            
        with torch.no_grad():
            self.token_prefix = word_embedding(torch.tensor(self.token_prefix).cuda())
            self.token_suffix = word_embedding(torch.tensor(self.token_suffix).cuda())
        
        self.attention_mask = torch.tensor(self.attention_mask).cuda()
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts, self.attention_mask

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer, cache_keys=None, cache_values=None):
        super().__init__()
        self.device = torch.device("cuda")
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, tokenizer)
        self.image_encoder = clip_model.visual
        self.zs_img_encoder = deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        self.image_features_store =[]
        self.cfg = cfg

        description_file = os.path.join('./description', f'{cfg.DATASET.NAME}.json')
        llm_descriptions = json.load(open(description_file))
        text_features =[]
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        
        oc_tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        for classname in classnames:
            prompts = []
            prompt = template.format(classname.replace("_", " "))
            prompts.append(prompt)

            for i in range(50):
                prompt_desc = prompt + ' ' + llm_descriptions[classname.replace("_", " ")][i]
                prompts.append(prompt_desc)
                
            prompts_tokens = oc_tokenizer(prompts, context_length=256).cuda()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    text_features.append(clip_model.encode_text(prompts_tokens)) 

        text_features = torch.cat(text_features) 
        _, d = text_features.shape
        self.ndisc = 51
        text_features = text_features.view(self.ndisc, -1, d)
        self.all_text_features_tea = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_mean = text_features.mean(dim=0)
        self.text_features_tea = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)
        self.text_prototypes = self.all_text_features_tea   

        if cfg.is_bonder:
            self.bonder = CrossAttnBlock(512).cuda()

        self.tip_adapter = None
        if cache_keys is not None:
            self.tip_alpha = 1.0  
            self.tip_beta = 5.5   
            self.tip_adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False).cuda()
            self.tip_adapter.weight = nn.Parameter(cache_keys) 
            self.register_buffer("cache_values", cache_values.cuda())

    def forward(self, image, mask=None, labels=None):
        with torch.no_grad():
            image_features_tea, local_image_features_tea, _ = self.zs_img_encoder(image)
            image_features_tea = image_features_tea / image_features_tea.norm(dim=-1, keepdim=True)
        
        image_features, local_image_features, _  = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

        prompts, attention_mask = self.prompt_learner()
        text_stu = self.text_encoder(prompts, attention_mask)
        text_stu = text_stu / text_stu.norm(dim=-1, keepdim=True)

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
            text_bias = text_bias / text_bias.norm(dim=-1, keepdim=True)
            alpha = self.cfg.lambda_value
            updated_proto = self.text_prototypes
            
            contra_labels = torch.arange(c).view(-1,1).cuda()
            mask = torch.eq(labels.unsqueeze(1), contra_labels.T).float().cuda()
            update_features = torch.matmul(mask.view(bs, c).transpose(0,1).unsqueeze(0).repeat(n_disc-1,1,1), text_bias.transpose(1, 0))
            proto_mask = torch.zeros(c, dtype=torch.int).cuda()
            proto_mask[labels] = 1
            proto_mask = proto_mask.view(1, -1, 1).repeat(n_disc, 1, d)
            update_features = torch.cat([self.text_prototypes[0:1, :, :], update_features], dim=0)
            updated_proto = (1-proto_mask) * updated_proto + proto_mask * (alpha * updated_proto + (1-alpha) * update_features)

            updated_proto_norm = updated_proto / updated_proto.norm(dim=-1, keepdim=True)
            updated_proto_mean = updated_proto_norm.mean(dim=0)
            updated_proto_mean_norm = updated_proto_mean / updated_proto_mean.norm(dim=-1, keepdim=True)
        else:
            updated_proto_norm = self.text_prototypes / self.text_prototypes.norm(dim=-1, keepdim=True)
            updated_proto_mean = updated_proto_norm.mean(dim=0)
            updated_proto_mean_norm = updated_proto_mean / updated_proto_mean.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * get_dense_logits2(image_features.detach(), local_image_features.detach(), updated_proto_norm, updated_proto_mean_norm, topk=self.cfg.topk)
        logits_local = logit_scale * get_dense_logits2(image_features, local_image_features, self.all_text_features_tea.detach(), self.text_features_tea.detach(), topk=self.cfg.topk)

        if self.tip_adapter is not None:
            affinity = self.tip_adapter(image_features)
            cache_logits = torch.exp(-self.tip_beta * (1.0 - affinity)) @ self.cache_values
            scaled_cache_logits = cache_logits * self.tip_alpha
            logits = logits + scaled_cache_logits
            logits_local = logits_local + scaled_cache_logits

        return logits, logits_local, image_features_tea, image_features, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea

@TRAINER_REGISTRY.register()
class LocProtoBiomed(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LOCOOP.PREC in["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.lambda_value = cfg.lambda_value
        self.top_k = cfg.topk
        self.label =[]

        # Load BiomedCLIP
        clip_model, _, tokenizer = load_biomedclip(self.device)

        if cfg.TRAINER.LOCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Extracting Pristine Visual Memory Cache from training set (Tip-Adapter)...")
        tfm_test = build_transform(cfg, is_train=False)
        cache_loader = build_data_loader(
            cfg,
            sampler_type="SequentialSampler",
            data_source=self.dm.dataset.train_x,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False
        )

        clip_model.to(self.device)
        clip_model.eval()
        
        cache_keys = []
        cache_labels =[]
        
        with torch.no_grad():
            for batch in tqdm(cache_loader, desc="Building Cache"):
                image = batch["img"].to(self.device)
                label = batch["label"].to(self.device)
                
                img_feat, _, _ = clip_model.visual(image)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                
                cache_keys.append(img_feat.cpu())
                cache_labels.append(label.cpu())
                
        cache_keys = torch.cat(cache_keys, dim=0).to(self.device) 
        cache_labels = torch.cat(cache_labels, dim=0).to(self.device) 
        cache_values = F.one_hot(cache_labels, num_classes=len(classnames)).float().to(self.device) 

        print("Building custom BiomedCLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, tokenizer, cache_keys=cache_keys, cache_values=cache_values)

        # ---------------- FIXED: Optimize timm blocks instead of OpenAI resblocks ----------------
        for name, param in self.model.named_parameters():
            if 'image_encoder.trunk.blocks.11.attn' in name or 'bonder' in name or 'tip_adapter' in name or 'prompt_learner' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        self.model.to(self.device)
        
        # Register Optimizers using TimmModel structure
        self.optim = build_optimizer(self.model.image_encoder.trunk.blocks[-1].attn, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("attn_learner", self.model.image_encoder.trunk.blocks[-1].attn, self.optim, self.sched)
        # -----------------------------------------------------------------------------------------
        
        cfg.OPTIM_PROMPT = deepcopy(cfg.OPTIM)
        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM_PROMPT)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM_PROMPT)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)

        if cfg.is_bonder:
            cfg.OPTIM2 = deepcopy(cfg.OPTIM)
            cfg.OPTIM2.LR = cfg.OPTIM.LR
            self.optim2 = build_optimizer(self.model.bonder, cfg.OPTIM2)
            self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM2)
            self.register_model("bonder_learner", self.model.bonder, self.optim2, self.sched2)

        if hasattr(self.model, "tip_adapter") and self.model.tip_adapter is not None:
            cfg.OPTIM_TIP = deepcopy(cfg.OPTIM)
            cfg.OPTIM_TIP.LR = 0.001
            self.optim_tip = build_optimizer(self.model.tip_adapter, cfg.OPTIM_TIP)
            self.sched_tip = build_lr_scheduler(self.optim_tip, cfg.OPTIM_TIP)
            self.register_model("tip_adapter_learner", self.model.tip_adapter, self.optim_tip, self.sched_tip)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LOCOOP.PREC

        if prec == "amp":
            with autocast():
                output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
                all_text_features_tea = self.model.all_text_features_tea.clone()
                
                loss_id = F.cross_entropy(output, label)
                loss_id2 = F.cross_entropy(output_local, label)
                loss_distil_img = F.l1_loss(img_feat_tea, img_feat_stu, reduction='mean') * 10
                
                if text_stu.shape == all_text_features_tea.shape:
                    loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
                else:
                    loss_distil_text = F.l1_loss(all_text_features_tea.mean(dim=0), text_stu.mean(dim=0), reduction='mean') * 25
                
                loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
                
                loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].zero_grad()
                    
            self.scaler.scale(loss).backward()
            
            for name in self._optims:
                if self._optims[name] is not None:
                    self.scaler.step(self._optims[name])
                    
            self.scaler.update()
        else:
            output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
            all_text_features_tea = self.model.all_text_features_tea.clone()
            
            loss_id = F.cross_entropy(output, label)
            loss_id2 = F.cross_entropy(output_local, label)
            loss_distil_img = F.l1_loss(img_feat_tea, img_feat_stu, reduction='mean') * 10
            
            if text_stu.shape == all_text_features_tea.shape:
                loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
            else:
                loss_distil_text = F.l1_loss(all_text_features_tea.mean(dim=0), text_stu.mean(dim=0), reduction='mean') * 25
                
            loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
            
            loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].zero_grad()
                    
            loss.backward()
            
            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].step()

        loss_summary = {
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_distil_img": loss_distil_img.item(),
            "loss_distil_text": loss_distil_text.item(),
            "acc": compute_accuracy(output_local, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        self.model.text_prototypes = self.model.all_text_features_tea.detach() 
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.val_loader if (split == "val" and self.val_loader is not None) else self.test_loader

        print(f"Evaluate on the *{split}* set")

        if self.cfg.is_bonder:
            self.model.text_prototypes = torch.load(osp.join(self.output_dir, 'proto.pth'))
            
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) >= 2:
                if self.cfg.is_bonder:
                    output = output[1] + 0.05 * output[0]
                else:
                    output = output[0]
            self.label.append(label)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        mcm_score =[]
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
            else:
                images = images.cuda()
            output, output_local, _, _, _, _, _, _, _ = self.model_inference(images)
            if self.cfg.is_bonder:
                output = output_local + 0.05 * output
            output /= 100.0
            smax_global = to_np(F.softmax(output/T, dim=-1))  
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_score.append(mcm_global_score)

        res = concat(mcm_score)[:len(data_loader.dataset)].copy()
        return res, res, res, res

    @torch.no_grad()
    def test_ood1(self, data_loader, T):
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score =[]
        mcm_score = []
        loc_score =[]
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
                labels = label
            else:
                images = images.cuda()
                labels = labels.cuda()
            
            output, output_local, _, _, _, _, _, _, _ = self.model_inference(images)
            
            batch_size, num_of_local_feature, _ = output_local.shape
            
            output /= 100.0
            output_local /= 100.0
            
            smax_global0 = to_np(F.softmax(output/T, dim=-1))
            smax_global = to_np(output)
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_global_score0 = -np.max(smax_global0, axis=1)
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            
            mcm_score.append(mcm_global_score)
            glmcm_score.append(mcm_global_score + mcm_local_score)
            loc_score.append(mcm_local_score)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), concat(loc_score)[:len(data_loader.dataset)].copy()