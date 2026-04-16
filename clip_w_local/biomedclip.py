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

    # Hook / Override the visual forward to return local features for TimmModel
    def new_visual_forward(self, x, mask=None):
        # 1. Forward through the timm trunk. For ViTs, this returns the unpooled sequence [B, 197, 768]
        features = self.trunk.forward_features(x)
        
        # 2. Extract Global (CLS token) and Local (Patch tokens)
        if hasattr(self, 'pool') and self.pool is not None:
            global_feat = self.pool(features)
        else:
            global_feat = features[:, 0, :]
            
        local_feat = features[:, 1:, :]
        
        # 3. Apply OpenCLIP's projection head (maps 768 -> 512 for BiomedCLIP)
        if hasattr(self, 'head') and isinstance(self.head, torch.nn.Linear):
            global_feat = self.head(global_feat)
            local_feat = self.head(local_feat) # nn.Linear broadcasts across the sequence dimension automatically
            
        return global_feat, local_feat, None

    # Replace the standard visual forward pass
    model.visual.forward = types.MethodType(new_visual_forward, model.visual)
    model = model.to(device)

    return model, preprocess_val, tokenizer