import pickle
import os

print("Checking model files...")
print(f"classifier.pkl exists: {os.path.exists('classifier.pkl')}")
print(f"fertilizer.pkl exists: {os.path.exists('fertilizer.pkl')}")

if os.path.exists('classifier.pkl') and os.path.exists('fertilizer.pkl'):
    try:
        with open('classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✓ classifier.pkl loaded successfully - Type: {type(model).__name__}")
        
        with open('fertilizer.pkl', 'rb') as f:
            ferti = pickle.load(f)
        print(f"✓ fertilizer.pkl loaded successfully - Type: {type(ferti).__name__}")
        
        # Test a prediction
        test_input = [[25, 78, 43, 4, 1, 22, 26, 38]]
        prediction = model.predict(test_input)[0]
        print(f"✓ Test prediction successful: class index {prediction}")
        
        if hasattr(ferti, "classes_"):
            fertilizer_name = ferti.classes_[prediction]
            print(f"✓ Fertilizer name: {fertilizer_name}")
        else:
            print("⚠ Warning: fertilizer encoder doesn't have classes_ attribute")
            
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("\nModels may be corrupted or incompatible. Regenerating models...")
        import subprocess
        subprocess.run(["python", "retrain.py"])
else:
    print("✗ Model files not found. Please run retrain.py to generate them.")
