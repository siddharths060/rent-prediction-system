import joblib

from preprocess import transform_input

# Load models and encoders
model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict(data: dict):
    df = transform_input(data,encoders)
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    sample_input = {
        "house_type": "3 BHK Flat for Rent in CV Raman Nagar, Bangalore",
        "locality": "C V Raman Nagar",
        "city": "Bangalore",
        "area": 1900.0,
        "beds": 3,
        "bathrooms": 3,
        "balconies": 3,
        "furnishing": "Semi-Furnished"
    }

    try:
        rent = predict(sample_input)
        print(f"\n🏠 Predicted Rent: ₹{int(rent)}")
    except Exception as e:
        print(f"❌ Error: {e}")