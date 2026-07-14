import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import os

class AdvancedForensicPipeline:
    """
    Unified Multi-Model Forensic Pipeline for Image & Document Forgery Detection.
    Intelligently routes assets and fuses spatial anomaly arrays based on model specialization.
    """
    def __init__(self, weights_dir: str = "weights/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "mps" if torch.backends.mps.is_available() 
                                   else "cpu")
        self.weights_dir = weights_dir
        print(f" Initializing Forensic Pipeline on Target Device: {self.device}")
        
        # In a full deployment, import your real model definitions:
        # from models.pscc import PSCCNet; from models.trufor import TruFor, etc.
        self._initialize_all_backbones()

    def _initialize_all_backbones(self):
        """Initializes and loads checkpoints for the specialized forensic variants."""
        # 1. PSCC-Net: Fast Generalist Baseline
        self.pscc_model = self._mock_load("pscc_net.pth")
        
        # 2. TruFor: High-Fidelity Multi-Modal Transformer Validator
        self.trufor_model = self._mock_load("trufor_heavy.pth")
        
        # 3. CAT-Net: JPEG Compression Grid Specialist
        self.catnet_model = self._mock_load("cat_net_dct.pth")
        
        # 4. BusterNet: Dedicated Copy-Move (Cloning) Detector
        self.busternet_model = self._mock_load("busternet_simi.pth")

    def _mock_load(self, weight_name: str) -> nn.Module:
        """Structural mock loader placeholder matching production forward pass requirements."""
        class MockModelShell(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
            def forward(self, x):
                # Returns spatial matrix matching input shape boundaries
                out = torch.sigmoid(self.conv(x))
                # Mock returning multi-scale lists for structural matching if required
                return [out, out, out, out] if "pscc" in weight_name else out
        
        model = MockModelShell().to(self.device)
        model.eval()
        return model

    def _preprocess(self, pil_image: Image.Image, target_size: int = 512) -> torch.Tensor:
        """Standardizes input shape grids and scales tensor value distributions."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        pipeline = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return pipeline(pil_image).unsqueeze(0).to(self.device)

    def process_asset(self, img_path: str, file_hint: str = "png") -> dict:
        """
        Processes an incoming asset through the multi-model forensic pipeline.
        Dynamically adjusts model fusion weights based on asset format constraints.
        """
        # Load asset safely into memory
        pil_img = Image.open(img_path)
        orig_w, orig_h = pil_img.size
        input_tensor = self._preprocess(pil_img)
        
        # Initialize an empty matrix tracking profile arrays
        heatmaps = {}
        
        with torch.no_grad():
            # ---- BRANCH 1: Execution of Fast General Baseline (PSCC-Net) ----
            pscc_out = self.pscc_model(input_tensor)[-1].squeeze().cpu().numpy()
            heatmaps['pscc'] = cv2.resize(pscc_out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            
            # ---- BRANCH 2: High Fidelity Verification Pass (TruFor) ----
            trufor_out = self.trufor_model(input_tensor).squeeze().cpu().numpy()
            heatmaps['trufor'] = cv2.resize(trufor_out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            
            # ---- BRANCH 3: Conditional Routing for JPEG Artifact Analysis (CAT-Net) ----
            if file_hint.lower() in ['jpg', 'jpeg']:
                cat_out = self.catnet_model(input_tensor).squeeze().cpu().numpy()
                heatmaps['catnet'] = cv2.resize(cat_out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            else:
                heatmaps['catnet'] = np.zeros((orig_h, orig_w)) # Zero weight allocation for PNGs

            # ---- BRANCH 4: Intra-Asset Patch Duplication Analysis (BusterNet) ----
            buster_out = self.busternet_model(input_tensor).squeeze().cpu().numpy()
            heatmaps['busternet'] = cv2.resize(buster_out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # ---- DYNAMIC INTELLIGENT FUSION LAYER ----
        # Allocate confidence weights based on architectural profile rules
        if file_hint.lower() in ['jpg', 'jpeg']:
            # Balance spatial structure, transformer analysis, and compression anomalies
            weights = {'pscc': 0.35, 'trufor': 0.35, 'catnet': 0.20, 'busternet': 0.10}
        else:
            # Drop compression grid weights entirely for native digital PNGs/Scans
            weights = {'pscc': 0.45, 'trufor': 0.45, 'catnet': 0.00, 'busternet': 0.10}

        # Compute the weighted linear combination of spatial anomaly channels
        master_heatmap = (
            weights['pscc'] * heatmaps['pscc'] +
            weights['trufor'] * heatmaps['trufor'] +
            weights['catnet'] * heatmaps['catnet'] +
            weights['busternet'] * heatmaps['busternet']
        )
        master_heatmap = np.clip(master_heatmap, 0.0, 1.0)
        
        # Derive quantitative core metrics
        peak_anomaly_score = float(np.max(master_heatmap))
        mean_anomaly_score = float(np.mean(master_heatmap))
        
        return {
            "master_heatmap": master_heatmap,
            "individual_heatmaps": heatmaps,
            "peak_anomaly": peak_anomaly_score,
            "mean_anomaly": mean_anomaly_score,
            "routing_profile": weights
        }