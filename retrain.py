import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the fertilizer dataset
try:
    data = pd.read_csv('f2.csv')
    
    # Encode categorical variables
    encode_soil = LabelEncoder()
    encode_crop = LabelEncoder()
    encode_ferti = LabelEncoder()
    
    data['Soil_Type'] = encode_soil.fit_transform(data['Soil_Type'])
    data['Crop_Type'] = encode_crop.fit_transform(data['Crop_Type'])
    data['Fertilizer'] = encode_ferti.fit_transform(data['Fertilizer'])
    
    # Split features and target
    X = data.drop('Fertilizer', axis=1)
    y = data['Fertilizer']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the fertilizer label encoder (used to map predictions back to fertilizer names)
    with open('fertilizer.pkl', 'wb') as f:
        pickle.dump(encode_ferti, f)
    
    print("Model training and saving complete.")
    print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
    
except FileNotFoundError:
    print("Error: f2.csv file not found. Please ensure the dataset file exists.")
except Exception as e:
    print(f"Error: {e}")
