"""
Claude-Powered Training Monitor
================================

Real-time training monitoring with AI insights from Claude
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.claude_assistant import ClaudeAssistant
import time
import json


class AITrainingMonitor:
    """
    Training monitor that uses Claude for insights
    """
    
    def __init__(self, check_every_n_epochs: int = 5):
        """
        Args:
            check_every_n_epochs: How often to ask Claude for analysis
        """
        try:
            self.assistant = ClaudeAssistant()
            self.enabled = True
            print("✓ Claude AI assistant initialized")
        except ValueError:
            print("⚠ Claude assistant not available (API key not set)")
            print("  Training will continue without AI insights")
            self.enabled = False
        
        self.check_every_n_epochs = check_every_n_epochs
        self.insights_history = []
    
    def on_epoch_end(self, epoch: int, metrics: dict):
        """
        Called at end of each epoch
        
        Args:
            epoch: Current epoch number
            metrics: Training metrics dictionary
        """
        if not self.enabled:
            return
        
        # Check if we should get Claude's analysis
        if epoch % self.check_every_n_epochs == 0 and epoch > 0:
            print(f"\n{'='*60}")
            print(f"Getting AI insights for epoch {epoch}...")
            print(f"{'='*60}")
            
            try:
                analysis = self.assistant.analyze_training_metrics(metrics, epoch)
                
                print("\n🤖 Claude's Analysis:")
                print("-" * 60)
                print(analysis)
                print("-" * 60)
                
                # Save insight
                self.insights_history.append({
                    'epoch': epoch,
                    'metrics': metrics,
                    'analysis': analysis,
                    'timestamp': time.time()
                })
                
                # Save to file
                self._save_insights()
                
            except Exception as e:
                print(f"⚠ Could not get AI insights: {e}")
    
    def _save_insights(self):
        """Save insights to JSON file"""
        insights_file = 'logs/claude_insights.json'
        os.makedirs('logs', exist_ok=True)
        
        with open(insights_file, 'w') as f:
            json.dump(self.insights_history, f, indent=2)
        
        print(f"✓ Insights saved to {insights_file}")
    
    def analyze_final_results(self, final_metrics: dict, model_summary: str):
        """
        Get final analysis and improvement suggestions
        
        Args:
            final_metrics: Final performance metrics
            model_summary: Summary of model architecture
        """
        if not self.enabled:
            return
        
        print(f"\n{'='*60}")
        print("Getting final AI recommendations...")
        print(f"{'='*60}")
        
        try:
            suggestions = self.assistant.suggest_improvements(
                model_summary, 
                final_metrics
            )
            
            print("\n🤖 Claude's Final Recommendations:")
            print("-" * 60)
            print(suggestions)
            print("-" * 60)
            
            # Save final analysis
            with open('logs/claude_final_analysis.txt', 'w') as f:
                f.write("Final Performance Metrics:\n")
                f.write(json.dumps(final_metrics, indent=2))
                f.write("\n\n" + "="*60 + "\n")
                f.write("Claude's Recommendations:\n")
                f.write("="*60 + "\n\n")
                f.write(suggestions)
            
            print("\n✓ Final analysis saved to logs/claude_final_analysis.txt")
            
        except Exception as e:
            print(f"⚠ Could not get final analysis: {e}")


# Example usage with training
if __name__ == "__main__":
    print("Claude-Powered Training Monitor Demo")
    print("="*60)
    
    # Initialize monitor
    monitor = AITrainingMonitor(check_every_n_epochs=5)
    
    # Simulate training
    for epoch in range(1, 16):
        # Simulated metrics
        metrics = {
            'train_loss': 0.5 - (epoch * 0.02),
            'val_loss': 0.55 - (epoch * 0.015),
            'miou': 0.60 + (epoch * 0.008),
            'lr': 0.001 * (0.95 ** epoch),
            'fps': 45 + (epoch * 0.5),
            'best_miou': max(0.60 + (epoch * 0.008), 0.72)
        }
        
        print(f"\nEpoch {epoch}: Loss={metrics['train_loss']:.3f}, mIoU={metrics['miou']:.3f}")
        
        # This would trigger Claude analysis every 5 epochs
        monitor.on_epoch_end(epoch, metrics)
    
    # Final analysis
    final_metrics = {
        'best_miou': 0.732,
        'final_loss': 0.21,
        'fps': 52,
        'parameters': '735K',
        'model_size_mb': 2.94
    }
    
    model_summary = """
FastSegNet - Efficient Real-Time Semantic Segmentation
- Encoder: MobileNetV2-style inverted residuals
- Decoder: Attention refinement + feature fusion
- Parameters: 735,219 (0.74M)
- Performance: 202 FPS on M4 Max GPU
"""
    
    monitor.analyze_final_results(final_metrics, model_summary)
