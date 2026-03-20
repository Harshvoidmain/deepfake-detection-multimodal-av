"""
Explainability tools for model interpretation (XAI)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanations
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, video, audio, class_idx=None):
        """
        Generate Grad-CAM visualization
        
        Args:
            video: Input video [B, T, C, H, W]
            audio: Input audio [B, audio_length]
            class_idx: Target class index
            
        Returns:
            cam: Attention map [B, T, H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(video, audio, return_features=True)
        logits = output['logits']
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[:, class_idx]
        class_score.backward(torch.ones_like(class_score))
        
        # Compute CAM
        gradients = self.gradients  # [B, C, ...]
        activations = self.activations  # [B, C, ...]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=list(range(2, len(gradients.shape))), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)  # [B, ...]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    @staticmethod
    def visualize(frame, cam, alpha=0.5):
        """
        Overlay CAM on original frame
        
        Args:
            frame: Original image [H, W, C]
            cam: Attention map [H, W]
            alpha: Blending factor
            
        Returns:
            visualization: Blended image [H, W, C]
        """
        # Resize CAM to frame size
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Blend with original frame
        if frame.max() <= 1.0:
            frame = frame.copy()
        else:
            frame = frame.astype(np.float32) / 255.0
            
        visualization = alpha * heatmap + (1 - alpha) * frame
        visualization = np.clip(visualization, 0, 1)
        
        return visualization


class ModelExplainer:
    """
    High-level explainability interface
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.gradcam = None
        
    def explain_prediction(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            video: Input video [1, T, C, H, W]
            audio: Input audio [1, audio_length]
            save_path: Path to save visualizations
            
        Returns:
            explanation: Dictionary with explanations and visualizations
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(video, audio, return_features=True)
            logits = output['logits']
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
        
        explanation = {
            'prediction': pred.item(),
            'confidence': probs[0, pred].item(),
            'probabilities': {
                'real': probs[0, 0].item(),
                'fake': probs[0, 1].item()
            },
            'reasoning': []
        }
        
        # Generate reasoning based on confidence
        if explanation['confidence'] > 0.9:
            explanation['reasoning'].append(f"High confidence ({explanation['confidence']:.2%}) in prediction")
        elif explanation['confidence'] > 0.7:
            explanation['reasoning'].append(f"Medium confidence ({explanation['confidence']:.2%}) in prediction")
        else:
            explanation['reasoning'].append(f"Low confidence ({explanation['confidence']:.2%}) - uncertain prediction")
        
        # Add feature-based reasoning (placeholder - can be enhanced)
        if 'features' in output:
            features = output['features']
            explanation['reasoning'].append(f"Visual features: {features['visual'].shape}")
            explanation['reasoning'].append(f"Audio features: {features['audio'].shape}")
            explanation['reasoning'].append(f"Frequency features: {features['frequency'].shape}")
        
        # Generate text explanation
        pred_label = "AI-generated/Fake" if pred.item() == 1 else "Real"
        explanation['text'] = f"""
        Prediction: {pred_label}
        Confidence: {explanation['confidence']:.2%}
        
        This video was classified as {pred_label.upper()} with {explanation['confidence']:.2%} confidence.
        
        Real probability: {probs[0, 0]:.2%}
        Fake probability: {probs[0, 1]:.2%}
        """
        
        return explanation
    
    def visualize_explanation(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        save_path: str
    ):
        """
        Create and save visual explanations
        
        Args:
            video: Input video [1, T, C, H, W]
            audio: Input audio [1, audio_length]
            save_path: Path to save visualization
        """
        explanation = self.explain_prediction(video, audio)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Prediction: {explanation['prediction']} | Confidence: {explanation['confidence']:.2%}")
        
        # Show sample frames
        frames = video[0].cpu().numpy()  # [T, C, H, W]
        frame_indices = np.linspace(0, len(frames)-1, 8, dtype=int)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(frame_indices):
                frame_idx = frame_indices[idx]
                frame = frames[frame_idx].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                ax.imshow(frame)
                ax.set_title(f"Frame {frame_idx}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {save_path}")
