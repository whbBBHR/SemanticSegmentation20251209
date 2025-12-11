# Claude AI Integration Guide

## Overview

This project includes **Claude AI assistant integration** for enhanced ML development workflow. Claude provides intelligent insights for training, annotation, and debugging.

## Features

### 1. **Training Insights** 📊
- Real-time analysis of training metrics
- Hyperparameter tuning suggestions
- Overfitting/underfitting detection
- Performance optimization recommendations

### 2. **Dataset Annotation** 🏷️
- AI-assisted image annotation using Claude's vision
- Object and region identification
- Challenging area detection
- Class label suggestions

### 3. **Error Analysis** 🔍
- Confusion matrix interpretation
- Common error pattern identification
- Architectural improvement suggestions
- Data augmentation recommendations

### 4. **Code Review** 💻
- Automated code review
- Optimization suggestions
- Bug detection
- Best practice recommendations

### 5. **Documentation** 📝
- Automatic documentation generation
- API reference creation
- Tutorial generation

### 6. **Research Assistance** 📚
- Latest research summarization
- Technique recommendations
- Benchmark comparisons

## Setup

### 1. Get API Key
```bash
# Sign up at https://console.anthropic.com/
# Get your API key from the dashboard
```

### 2. Install Package
```bash
pip install anthropic
```

### 3. Set Environment Variable
```bash
# Add to your ~/.zshrc or ~/.bashrc
export ANTHROPIC_API_KEY='your-api-key-here'

# Or set for current session
export ANTHROPIC_API_KEY='sk-ant-...'
```

### 4. Verify Setup
```python
from src.utils.claude_assistant import ClaudeAssistant

assistant = ClaudeAssistant()
print("✓ Claude assistant ready!")
```

## Usage Examples

### Training Monitor

```python
from examples.claude_training_monitor import AITrainingMonitor

# Initialize monitor (checks every 5 epochs)
monitor = AITrainingMonitor(check_every_n_epochs=5)

# In your training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    metrics = {
        'train_loss': loss,
        'val_loss': val_loss,
        'miou': miou,
        'lr': current_lr,
        'fps': fps
    }
    
    # Get AI insights
    monitor.on_epoch_end(epoch, metrics)

# Final analysis
monitor.analyze_final_results(final_metrics, model_summary)
```

### Dataset Annotation

```bash
# Annotate directory of images
python examples/claude_annotation_helper.py annotate \
    --image-dir data/raw/images/ \
    --output annotations.txt

# Explain a prediction
python examples/claude_annotation_helper.py explain \
    --image test.jpg \
    --predictions 'road,sidewalk,building,sky'
```

### Direct API Usage

```python
from src.utils.claude_assistant import ClaudeAssistant

assistant = ClaudeAssistant()

# Analyze training metrics
metrics = {'train_loss': 0.45, 'miou': 0.68, 'lr': 0.001}
analysis = assistant.analyze_training_metrics(metrics, epoch=25)
print(analysis)

# Help with image annotation
analysis = assistant.analyze_image_for_annotation('image.jpg')
print(analysis)

# Explain segmentation result
explanation = assistant.explain_segmentation_result(
    'image.jpg', 
    'Detected: road, cars, buildings, sky'
)
print(explanation)

# Get research summary
summary = assistant.research_summary('real-time semantic segmentation')
print(summary)

# Code review
review = assistant.review_code('src/models/efficient_segmentation.py')
print(review)
```

## Cost Estimation

Claude API pricing (as of Dec 2025):
- **Input**: $3 per million tokens (~750K words)
- **Output**: $15 per million tokens
- **Vision**: $3 per 1000 images (input)

**Typical costs:**
- Training monitoring (every 5 epochs, 100 epochs): ~$0.50
- Dataset annotation (100 images): ~$0.30
- Code review (5 files): ~$0.10
- **Total for full project**: ~$1-2

## Best Practices

### ✅ DO Use Claude For:
- **Development phase** - Training insights, debugging
- **Dataset preparation** - Annotation assistance
- **Code review** - Before important commits
- **Research** - Understanding new techniques
- **Documentation** - Auto-generating docs

### ❌ DON'T Use Claude For:
- **Production inference** - Too slow, adds latency
- **Real-time applications** - Use your trained model
- **High-frequency calls** - Costs add up
- **Training the model** - Claude doesn't train neural nets

## Examples Output

### Training Analysis
```
🤖 Claude's Analysis:
------------------------------------------------------------
Your model shows healthy convergence with validation loss 
tracking training loss closely. The mIoU of 0.68 at epoch 25
is progressing well toward your target.

Recommendations:
1. Consider reducing learning rate by 2x at epoch 30
2. Add more data augmentation for road/sidewalk confusion
3. Current FPS of 45 is excellent for real-time use
4. Monitor for plateau after epoch 40 - may need warmup restart
------------------------------------------------------------
```

### Annotation Help
```
🤖 Image Analysis:
------------------------------------------------------------
Detected regions for segmentation:
1. Road (bottom 40%) - asphalt, clear boundary
2. Sidewalk (left edge) - concrete, partially occluded by parked car
3. Building (center-right) - brick facade with windows
4. Sky (top 25%) - clear blue, easy to segment
5. Vehicles - 2 cars (challenging: one partially cropped)
6. Vegetation - tree on left (irregular boundary - challenging)

Suggested attention areas:
- Car-road boundary (shadow makes it difficult)
- Building-sky edge (windows may confuse the model)
------------------------------------------------------------
```

## Troubleshooting

### API Key Not Found
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set in current session
export ANTHROPIC_API_KEY='your-key'

# Or pass directly
assistant = ClaudeAssistant(api_key='your-key')
```

### Rate Limits
Claude has rate limits:
- **Tier 1**: 50 requests/minute
- **Tier 2**: 1000 requests/minute

Add delays if hitting limits:
```python
import time
time.sleep(1)  # Wait between requests
```

### Image Format Issues
Supported formats: JPG, PNG, GIF, WebP

Convert if needed:
```python
from PIL import Image
img = Image.open('image.bmp')
img.save('image.jpg')
```

## Integration with Training

To integrate with your training pipeline:

```python
# In src/training/training_pipeline.py

from src.utils.claude_assistant import ClaudeAssistant

class SegmentationTrainer:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add Claude assistant (optional)
        try:
            self.ai_assistant = ClaudeAssistant()
            self.use_ai_insights = True
        except:
            self.use_ai_insights = False
    
    def train_epoch(self, epoch):
        # ... training code ...
        
        # Get AI insights every N epochs
        if self.use_ai_insights and epoch % 5 == 0:
            analysis = self.ai_assistant.analyze_training_metrics(
                self.get_metrics(), epoch
            )
            print(f"\n🤖 AI Insights:\n{analysis}\n")
```

## See Also

- [Anthropic Documentation](https://docs.anthropic.com/)
- [Claude API Reference](https://docs.anthropic.com/claude/reference/)
- [Vision API Guide](https://docs.anthropic.com/claude/docs/vision)

---

**Note**: Claude integration is **optional**. The project works perfectly without it - Claude just adds intelligent assistance during development! 🤖✨
