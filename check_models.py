#!/usr/bin/env python
import pickle
import os
import sys

print("=" * 50)
print("MODEL FILE CHECKER")
print("=" * 50)

# Check if files exist
classifier_exists = os.path.exists('classifier.pkl')
fertilizer_exists = os.path.exists('fertilizer.pkl')

print(f"\n1. File Existence Check:")
print(f"   classifier.pkl: {'✓ EXISTS' if classifier_exists else '✗ NOT FOUND'}")
print(f"   fertilizer.pkl: {'✓ EXISTS' if fertilizer_exists else '✗ NOT FOUND'}")

if not classifier_exists or not fertilizer_exists:
    print("\n❌ ERROR: Model files are missing!")
    print("   Solution: Run 'python retrain.py' to generate the models.")
    sys.exit(1)

print("\n2. Loading Models...")
try:
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ classifier.pkl loaded - Type: {type(model).__name__}")
    
    with open('fertilizer.pkl', 'rb') as f:
        ferti = pickle.load(f)
    print(f"   ✓ fertilizer.pkl loaded - Type: {type(ferti).__name__}")
except Exception as e:
    print(f"   ✗ ERROR loading models: {e}")
    print("\n   The model files may be corrupted or incompatible.")
    print("   Solution: Run 'python retrain.py' to regenerate the models.")
    sys.exit(1)

print("\n3. Testing Prediction...")
try:
    test_input = [[25, 78, 43, 4, 1, 22, 26, 38]]
    prediction = model.predict(test_input)[0]
    print(f"   ✓ Prediction successful - Class index: {prediction}")
    
    if hasattr(ferti, "classes_"):
        fertilizer_name = ferti.classes_[prediction]
        print(f"   ✓ Fertilizer name: {fertilizer_name}")
        print(f"   ✓ Total fertilizer types: {len(ferti.classes_)}")
    else:
        print("   ⚠ Warning: fertilizer encoder doesn't have 'classes_' attribute")
except Exception as e:
    print(f"   ✗ ERROR during prediction: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ ALL CHECKS PASSED! Models are ready to use.")
print("=" * 50)
print("\nYou can now run: python main.py")
