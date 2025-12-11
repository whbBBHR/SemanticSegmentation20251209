"""
Comprehensive Test Suite for Claude Integration
================================================
Tests all Claude assistant features to ensure consistency
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.claude_assistant import ClaudeAssistant
from src.utils.load_env import load_env

def test_initialization():
    """Test 1: Claude assistant initialization"""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    load_env()
    
    try:
        assistant = ClaudeAssistant()
        print(f"✅ Claude assistant initialized")
        print(f"   Model: {assistant.model}")
        print(f"   API key length: {len(assistant.api_key)}")
        return assistant, True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None, False


def test_training_metrics(assistant):
    """Test 2: Training metrics analysis"""
    print("\n" + "="*60)
    print("TEST 2: Training Metrics Analysis")
    print("="*60)
    
    metrics = {
        'train_loss': 0.35,
        'val_loss': 0.42,
        'train_miou': 0.68,
        'val_miou': 0.65,
        'learning_rate': 0.0005,
        'best_miou': 0.70
    }
    
    try:
        analysis = assistant.analyze_training_metrics(metrics, epoch=10)
        print(f"✅ Training analysis successful")
        print(f"   Response length: {len(analysis)} chars")
        print(f"   Preview: {analysis[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Training analysis failed: {e}")
        return False


def test_image_annotation(assistant):
    """Test 3: Image annotation"""
    print("\n" + "="*60)
    print("TEST 3: Image Annotation")
    print("="*60)
    
    test_image = "data/test_images/urban_scene.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠ Test image not found: {test_image}")
        return False
    
    try:
        analysis = assistant.analyze_image_for_annotation(
            test_image,
            task_description="Semantic segmentation for urban scenes"
        )
        print(f"✅ Image annotation successful")
        print(f"   Response length: {len(analysis)} chars")
        print(f"   Preview: {analysis[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Image annotation failed: {e}")
        return False


def test_segmentation_explanation(assistant):
    """Test 4: Segmentation result explanation"""
    print("\n" + "="*60)
    print("TEST 4: Segmentation Explanation")
    print("="*60)
    
    test_image = "data/test_images/urban_scene.jpg"
    prediction = "Detected: sky (35%), road (25%), building (30%), car (10%)"
    
    if not os.path.exists(test_image):
        print(f"⚠ Test image not found: {test_image}")
        return False
    
    try:
        explanation = assistant.explain_segmentation_result(test_image, prediction)
        print(f"✅ Segmentation explanation successful")
        print(f"   Response length: {len(explanation)} chars")
        print(f"   Preview: {explanation[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Segmentation explanation failed: {e}")
        return False


def test_error_analysis(assistant):
    """Test 5: Error pattern analysis"""
    print("\n" + "="*60)
    print("TEST 5: Error Pattern Analysis")
    print("="*60)
    
    confusion = {
        'road_as_sidewalk': 145,
        'car_as_truck': 23,
        'building_as_sky': 12,
        'person_as_background': 8
    }
    
    try:
        analysis = assistant.analyze_error_patterns(confusion, val_miou=0.65)
        print(f"✅ Error analysis successful")
        print(f"   Response length: {len(analysis)} chars")
        print(f"   Preview: {analysis[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Error analysis failed: {e}")
        return False


def test_code_review(assistant):
    """Test 6: Code review"""
    print("\n" + "="*60)
    print("TEST 6: Code Review")
    print("="*60)
    
    code = """
def train_model(model, dataloader, epochs=10):
    for epoch in range(epochs):
        loss = 0
        for batch in dataloader:
            output = model(batch)
            loss += output.loss
        print(f'Epoch {epoch}: {loss}')
"""
    
    try:
        review = assistant.review_code(code, context="Training loop for segmentation")
        print(f"✅ Code review successful")
        print(f"   Response length: {len(review)} chars")
        print(f"   Preview: {review[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Code review failed: {e}")
        return False


def test_documentation(assistant):
    """Test 7: Documentation generation"""
    print("\n" + "="*60)
    print("TEST 7: Documentation Generation")
    print("="*60)
    
    code = """
class FastSegNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.encoder = MobileNetV2()
        self.decoder = LightweightDecoder()
    
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
"""
    
    try:
        docs = assistant.generate_documentation(code, style="docstring")
        print(f"✅ Documentation generation successful")
        print(f"   Response length: {len(docs)} chars")
        print(f"   Preview: {docs[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Documentation generation failed: {e}")
        return False


def test_research_summary(assistant):
    """Test 8: Research summary"""
    print("\n" + "="*60)
    print("TEST 8: Research Summary")
    print("="*60)
    
    try:
        summary = assistant.research_summary(
            topic="real-time semantic segmentation with MobileNet architectures",
            focus="efficiency and accuracy tradeoffs"
        )
        print(f"✅ Research summary successful")
        print(f"   Response length: {len(summary)} chars")
        print(f"   Preview: {summary[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Research summary failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "🧪 CLAUDE INTEGRATION TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Initialization
    assistant, results['init'] = test_initialization()
    
    if not assistant:
        print("\n❌ Cannot proceed without valid assistant")
        return
    
    # Test 2-8: All other features
    results['training_metrics'] = test_training_metrics(assistant)
    results['image_annotation'] = test_image_annotation(assistant)
    results['segmentation_explain'] = test_segmentation_explanation(assistant)
    results['error_analysis'] = test_error_analysis(assistant)
    results['code_review'] = test_code_review(assistant)
    results['documentation'] = test_documentation(assistant)
    results['research_summary'] = test_research_summary(assistant)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Claude integration is working perfectly.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review errors above.")


if __name__ == "__main__":
    run_all_tests()
