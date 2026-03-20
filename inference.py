"""
Inference script for deepfake video detection
"""

import torch
import torch.nn.functional as F
import os
import argparse
from pathlib import Path

from models import MultimodalAVDetector
from utils import (
    DeepfakeDataset,
    get_transforms,
    load_config,
    load_checkpoint,
    get_device,
    ModelExplainer
)


class Predictor:
    """
    Predictor class for inference on new videos
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        # Load configuration
        self.config = load_config(config_path)
        
        # Get device
        self.device = get_device(self.config)
        
        # Initialize model
        self.model = MultimodalAVDetector(self.config).to(self.device)
        
        # Load checkpoint
        load_checkpoint(self.model, checkpoint_path, device=self.device)
        self.model.eval()
        
        # Initialize explainer
        if self.config['explainability']['enabled']:
            self.explainer = ModelExplainer(self.model, self.config)
        else:
            self.explainer = None
        
        print("Predictor initialized successfully")
        
    @torch.no_grad()
    def predict_single(self, video_path: str, return_explanation: bool = False):
        """
        Predict on a single video
        
        Args:
            video_path: Path to video file
            return_explanation: If True, return detailed explanation
            
        Returns:
            result: Dictionary with prediction results
        """
        # Load video and audio
        # For simplicity, using DeepfakeDataset's methods
        # In production, create a separate VideoLoader class
        
        # Create dummy dataset with single sample
        import json
        import tempfile
        
        # Create temporary metadata
        temp_metadata = {
            'test': [{
                'path': video_path,
                'label': 0  # Dummy label
            }]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(temp_metadata, f)
            temp_metadata_path = f.name
        
        try:
            # Get transforms
            _, val_transforms = get_transforms(self.config, mode='test')
            
            # Create dataset
            dataset = DeepfakeDataset(
                data_dir='',  # Not used since we have full path
                metadata_file=temp_metadata_path,
                num_frames=self.config['data']['num_frames'],
                frame_size=self.config['data']['frame_size'],
                audio_sample_rate=self.config['data']['audio_sample_rate'],
                clip_duration=self.config['data']['clip_duration'],
                transform=val_transforms[0],
                audio_transform=val_transforms[1],
                mode='test'
            )
            
            # Get sample
            sample = dataset[0]
            video = sample['video'].unsqueeze(0).to(self.device)  # [1, T, C, H, W]
            audio = sample['audio'].unsqueeze(0).to(self.device)  # [1, audio_length]
            
        finally:
            # Clean up temp file
            os.unlink(temp_metadata_path)
        
        # Prediction
        output = self.model(video, audio)
        logits = output['logits']
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        
        result = {
            'video_path': video_path,
            'prediction': 'Fake' if pred.item() == 1 else 'Real',
            'confidence': probs[0, pred].item(),
            'probabilities': {
                'real': probs[0, 0].item(),
                'fake': probs[0, 1].item()
            }
        }
        
        # Generate explanation if requested
        if return_explanation and self.explainer is not None:
            explanation = self.explainer.explain_prediction(video, audio)
            result['explanation'] = explanation
            
            # Save visualization
            if self.config['explainability']['save_visualizations']:
                viz_dir = self.config['explainability']['visualization_dir']
                os.makedirs(viz_dir, exist_ok=True)
                
                video_name = Path(video_path).stem
                viz_path = os.path.join(viz_dir, f'{video_name}_explanation.png')
                self.explainer.visualize_explanation(video, audio, viz_path)
                result['visualization_path'] = viz_path
        
        return result
    
    def predict_batch(self, video_dir: str, output_file: str = None):
        """
        Predict on all videos in a directory
        
        Args:
            video_dir: Directory containing video files
            output_file: Optional path to save results as JSON
            
        Returns:
            results: List of prediction results
        """
        video_extensions = self.config['data']['video_extensions']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        print(f"Found {len(video_files)} videos in {video_dir}")
        
        results = []
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            try:
                result = self.predict_single(str(video_file))
                results.append(result)
                
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Real: {result['probabilities']['real']:.2%} | Fake: {result['probabilities']['fake']:.2%}")
                
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                continue
        
        # Save results if output file specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Deepfake video detection inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to single video file')
    parser.add_argument('--video_dir', type=str, default=None,
                       help='Path to directory containing videos')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for batch predictions')
    parser.add_argument('--explain', action='store_true',
                       help='Generate explanations')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = Predictor(args.config, args.checkpoint)
    
    # Single video prediction
    if args.video:
        result = predictor.predict_single(args.video, return_explanation=args.explain)
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Video: {result['video_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Real: {result['probabilities']['real']:.2%}")
        print(f"  Fake: {result['probabilities']['fake']:.2%}")
        
        if 'explanation' in result:
            print(f"\n{result['explanation']['text']}")
        
        if 'visualization_path' in result:
            print(f"\nVisualization saved to: {result['visualization_path']}")
        
        print("="*60)
    
    # Batch prediction
    elif args.video_dir:
        results = predictor.predict_batch(args.video_dir, args.output)
        
        # Summary statistics
        total = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'Fake')
        real_count = total - fake_count
        
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total videos: {total}")
        print(f"Predicted Real: {real_count} ({real_count/total*100:.1f}%)")
        print(f"Predicted Fake: {fake_count} ({fake_count/total*100:.1f}%)")
        print("="*60)
    
    else:
        print("Error: Please specify either --video or --video_dir")


if __name__ == '__main__':
    main()
