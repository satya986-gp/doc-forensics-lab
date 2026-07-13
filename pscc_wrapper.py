import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

class PSCCNetInferenceEngine:
    """
    Inference wrapper engine for the Progressive Spatio-Channel Correlation Network (PSCC-Net).
    Handles preprocessing, device distribution, forward passing, and mask upsampling.
    """
    def __init__(self, model_weights_path: str = "weights/pscc_net_checkpoint.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "mps" if torch.backends.mps.is_available() 
                                   else "cpu")
        self.weights_path = model_weights_path
        self.model = self._initialize_network()
        
    def _initialize_network(self) -> nn.Module:
        """
        Initializes the model structure. 
        Note: Swap this structural placeholder out with your precise repo imports 
        (e.g., from models.pscc import PSCCNet) once you match the repository layout.
        """
        # Placeholder dummy shell matching PSCC architecture inputs/outputs
        class PSCCStructureMock(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_layer = nn.Conv2d(3, 1, kernel_size=3, padding=1)
            def forward(self, x):
                # PSCC-Net progressively estimates 4 masks at different scales
                # We return a list of predictions; the final one [-1] is the highest resolution (1/1)
                mask_coarse = torch.sigmoid(self.dummy_layer(x))
                mask_fine = mask_coarse.clone() 
                return [mask_coarse, mask_coarse, mask_coarse, mask_fine]
        
        model = PSCCStructureMock()
        
        # Load pre-trained weights safely if present
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded PSCC-Net weights successfully from: {self.weights_path}")
        except Exception as e:
            print(f"⚠️ Weights file not found at {self.weights_path}. Running with initialized base states. ({e})")
            
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, pil_image: Image.Image, target_size: int = 512):
        """Converts PIL image to normal tracking shapes and transforms into input tensors."""
        # Convert image to RGB format if uploaded as grayscale/RGBA
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        transform_pipeline = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform_pipeline(pil_image).unsqueeze(0) # Add batch axis
        return input_tensor.to(self.device)

    def predict_tamper_mask(self, pil_image: Image.Image, threshold: float = 0.5) -> tuple:
        """
        Runs inference on input PIL Image.
        Returns:
            - binary_mask (np.ndarray): Boolean mask scaled back to original image size
            - heatmap (np.ndarray): Soft probability mask scaled back to original image size
        """
        orig_w, orig_h = pil_image.size
        
        # Preprocess and execute forward pass without gradient calculations
        input_tensor = self._preprocess(pil_image)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # PSCC-Net outputs a list of 4 scale masks. The last one is the target high-res mask
            final_mask_tensor = outputs[-1].squeeze(0).squeeze(0) # Strip batch and channel dimensions
            
        # Push matrix tensor to CPU numpy tracking format
        heatmap = final_mask_tensor.cpu().numpy()
        
        # Upsample back to match original viewport resolution layout cleanly
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        
        # Binary threshold calculation
        binary_mask = (heatmap_resized > threshold).astype(np.uint8) * 255
        
        return binary_mask, heatmap_resized