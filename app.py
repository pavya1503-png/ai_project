from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from textblob import TextBlob

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ SENTIMENT PREDICTION
@app.get("/predict")
def predict(text: str):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    # convert polarity to simple prediction (0 or 1)
    prediction = 1 if polarity > 0 else 0

    return {
        "prediction": prediction,
        "polarity": polarity
    }

# ✅ SIMPLE GENERATE ENDPOINT
@app.get("/generate")
def generate(text: str):
    return {"response": f"AI says: {text} is interesting!"}

# ✅ TRITON SIMULATION
@app.get("/triton")
def triton():
    return {"message": "Triton inference simulated successfully"}