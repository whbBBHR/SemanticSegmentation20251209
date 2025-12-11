"""
Claude AI Assistant Integration
================================

Uses Anthropic's Claude API for:
- Dataset annotation assistance
- Training optimization suggestions
- Error analysis and debugging
- Code documentation generation
- Model performance analysis
"""

import anthropic
import base64
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

# Try to load .env file if it exists
try:
    from .load_env import load_env
    load_env()
except:
    pass


class ClaudeAssistant:
    """
    Integration with Claude AI for various ML development tasks
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude assistant
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-haiku-20241022"  # Fast, cost-effective model
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    
    def analyze_training_metrics(self, metrics: Dict[str, Any], epoch: int) -> str:
        """
        Analyze training metrics and provide optimization suggestions
        
        Args:
            metrics: Dictionary with training metrics (loss, mIoU, lr, etc.)
            epoch: Current training epoch
            
        Returns:
            Claude's analysis and suggestions
        """
        prompt = f"""
Analyze these semantic segmentation training metrics from epoch {epoch}:

Current Performance:
- Training Loss: {metrics.get('train_loss', 'N/A')}
- Validation Loss: {metrics.get('val_loss', 'N/A')}
- Mean IoU: {metrics.get('miou', 'N/A')}
- Learning Rate: {metrics.get('lr', 'N/A')}
- FPS: {metrics.get('fps', 'N/A')}

Previous Best mIoU: {metrics.get('best_miou', 'N/A')}

Please provide:
1. Assessment of current training progress
2. Potential issues (overfitting, underfitting, plateaus)
3. Specific hyperparameter adjustment suggestions
4. Data augmentation recommendations if needed
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def analyze_image_for_annotation(self, image_path: str, task_description: str = "") -> str:
        """
        Use Claude's vision to help with dataset annotation
        
        Args:
            image_path: Path to image file
            task_description: Optional description of what to look for
            
        Returns:
            Claude's description of objects/regions for labeling
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        base_prompt = """Analyze this image for semantic segmentation annotation. Identify and describe:
1. All distinct objects and their boundaries
2. Different regions (road, sidewalk, building, sky, vegetation, etc.)
3. Challenging areas that might be hard to segment
4. Suggested class labels for each region"""
        
        prompt = f"{base_prompt}\n\n{task_description}" if task_description else base_prompt
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": self.encode_image(image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return response.content[0].text
    
    def explain_segmentation_result(self, image_path: str, prediction_summary: str) -> str:
        """
        Explain a segmentation result to help understand model behavior
        
        Args:
            image_path: Path to original image
            prediction_summary: Summary of predicted classes/regions
            
        Returns:
            Claude's explanation of the segmentation
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        prompt = f"""
This image was processed by a semantic segmentation model with these results:
{prediction_summary}

Please:
1. Describe what you see in the image
2. Assess if the segmentation results seem accurate
3. Identify any potential errors or misclassifications
4. Suggest why certain areas might have been challenging for the model
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": self.encode_image(image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return response.content[0].text
    
    def analyze_error_patterns(self, confusion_matrix: str, common_errors: List[str]) -> str:
        """
        Analyze segmentation errors and suggest improvements
        
        Args:
            confusion_matrix: String representation of confusion matrix
            common_errors: List of common error descriptions
            
        Returns:
            Claude's analysis and suggestions
        """
        errors_text = "\n".join([f"- {error}" for error in common_errors])
        
        prompt = f"""
Analyze these semantic segmentation errors:

Confusion Matrix:
{confusion_matrix}

Common Error Patterns:
{errors_text}

Please provide:
1. Which classes are most confused with each other
2. Possible reasons for these confusions
3. Architectural improvements to address these issues
4. Data augmentation strategies
5. Post-processing techniques that might help
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def review_code(self, code_path: str, review_focus: str = "optimization") -> str:
        """
        Get code review and suggestions from Claude
        
        Args:
            code_path: Path to Python file
            review_focus: What to focus on (optimization, bugs, documentation, etc.)
            
        Returns:
            Claude's code review
        """
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"Code file not found: {code_path}")
        
        with open(code_path, 'r') as f:
            code = f.read()
        
        prompt = f"""
Review this Python code for semantic segmentation with focus on {review_focus}:

```python
{code}
```

Provide:
1. Overall assessment
2. Specific improvements for {review_focus}
3. Potential bugs or issues
4. Best practice recommendations
5. Performance optimization suggestions
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_documentation(self, code_path: str, doc_type: str = "comprehensive") -> str:
        """
        Generate documentation for code
        
        Args:
            code_path: Path to Python file
            doc_type: Type of documentation (comprehensive, api, tutorial)
            
        Returns:
            Generated documentation
        """
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"Code file not found: {code_path}")
        
        with open(code_path, 'r') as f:
            code = f.read()
        
        prompt = f"""
Generate {doc_type} documentation for this Python code:

```python
{code}
```

Include:
1. Overview of functionality
2. Class/function descriptions with parameters and return values
3. Usage examples
4. Architecture explanations where relevant
5. Performance characteristics
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def research_summary(self, topic: str) -> str:
        """
        Get summary of recent research on a topic
        
        Args:
            topic: Research topic (e.g., "real-time semantic segmentation")
            
        Returns:
            Research summary
        """
        prompt = f"""
Summarize recent advances and state-of-the-art in {topic}:

1. Key recent papers and techniques
2. Performance benchmarks on standard datasets
3. Novel architectures or approaches
4. Optimization strategies for deployment
5. Current challenges and future directions

Focus on practical techniques that can improve model performance and efficiency.
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def suggest_improvements(self, model_summary: str, performance_metrics: Dict) -> str:
        """
        Get improvement suggestions based on current model performance
        
        Args:
            model_summary: Description of current model architecture
            performance_metrics: Current performance numbers
            
        Returns:
            Improvement suggestions
        """
        prompt = f"""
Model Architecture:
{model_summary}

Current Performance:
{json.dumps(performance_metrics, indent=2)}

Suggest improvements to:
1. Increase accuracy (mIoU)
2. Improve inference speed (FPS)
3. Reduce model size
4. Handle edge cases better
5. Improve training stability

Prioritize suggestions that maintain real-time performance (>30 FPS).
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text


# Example usage
if __name__ == "__main__":
    # Initialize assistant (requires ANTHROPIC_API_KEY environment variable)
    try:
        assistant = ClaudeAssistant()
        
        # Example: Analyze training metrics
        metrics = {
            'train_loss': 0.45,
            'val_loss': 0.52,
            'miou': 0.68,
            'lr': 0.001,
            'fps': 45,
            'best_miou': 0.72
        }
        
        print("Analyzing training metrics...")
        analysis = assistant.analyze_training_metrics(metrics, epoch=25)
        print(analysis)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use Claude integration:")
        print("1. Get API key from https://console.anthropic.com/")
        print("2. Set environment variable: export ANTHROPIC_API_KEY='your-key'")
        print("3. Or pass api_key parameter to ClaudeAssistant()")
