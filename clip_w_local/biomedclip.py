import torch
import open_clip
from transformers import AutoTokenizer
import types

def load_biomedclip(device="cuda"):
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    print(f"Loading BiomedCLIP from {model_name}...")
    
    # Load model and transforms
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    # Hook / Override the visual forward to return local features
    def new_visual_forward(self, x, mask=None):
        # OpenCLIP ViT forward
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Extract Global and Local features
        global_feat = self.ln_post(x[:, 0, :])
        local_feat = self.ln_post(x[:, 1:, :])
        
        if self.proj is not None:
            global_feat = global_feat @ self.proj
            local_feat = local_feat @ self.proj
            
        return global_feat, local_feat, None

    # Replace the standard visual forward pass
    model.visual.forward = types.MethodType(new_visual_forward, model.visual)
    model = model.to(device)

    return model, preprocess_val, tokenizer