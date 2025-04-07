from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import torch
from pathlib import Path
import uvicorn
import sys
import yaml
from nlp_feature_mapper import translate_to_features  # ✅ NLP feature mapping

# ---- Threat label mapping (22 classes) ----
threat_labels = {
    0: "Normal Traffic",
    1: "DoS: Back",
    2: "DoS: Land",
    3: "DoS: Neptune",
    4: "DoS: Pod",
    5: "DoS: Smurf",
    6: "DoS: Teardrop",
    7: "Probe: Satan",
    8: "Probe: Ipsweep",
    9: "Probe: Nmap",
    10: "Probe: Portsweep",
    11: "R2L: Guess Password",
    12: "R2L: FTP Write",
    13: "R2L: IMAP",
    14: "R2L: PHF",
    15: "R2L: Multihop",
    16: "R2L: Warezmaster",
    17: "R2L: Warezclient",
    18: "U2R: Buffer Overflow",
    19: "U2R: Loadmodule",
    20: "U2R: Perl",
    21: "U2R: Rootkit",
    22: "Suspicious or Unknown Behavior"
}

# ---- Add model source path ----
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.base.network_analyzer import NetworkAnalyzer  # Adjust if needed

# ---- Setup ----
app = FastAPI()

# ---- Request Schemas ----
class Query(BaseModel):
    features: list  # Expecting a list of numerical features

class NLPMessage(BaseModel):
    message: str  # Natural language message

# ---- Load Model and Config ----
model_path = project_root / "models" / "best_model.pth"
config_path = project_root / "src" / "config" / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

expected_input_size = config['model']['network_analyzer']['input_size']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NetworkAnalyzer(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---- /predict Route ----
@app.post("/predict")
def predict(query: Query):
    features = query.features
    print("🔍 Received features:", features)

    if not isinstance(features, list) or len(features) != expected_input_size:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_input_size} features, but got {len(features)}."
        )

    try:
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            predicted = torch.argmax(output, dim=1).item()
            label = threat_labels.get(predicted, "Unknown")
            confidence = round(float(probs[predicted]) * 100, 2)

            print(f"✅ Predicted class: {predicted} ({label}) | Confidence: {confidence}%")

        return {
            "prediction": predicted,
            "label": label,
            "confidence": f"{confidence}%"
        }

    except Exception as e:
        print("❌ Inference error:", str(e))
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

# ---- /predict-nlp Route (with fallback logic) ----
@app.post("/predict-nlp")
def predict_nlp(input_msg: NLPMessage):
    try:
        text = input_msg.message.strip()
        features = translate_to_features(text)

        print("🧠 Input:", text)
        print("🔎 Extracted Features:", features)

        if len(features) != expected_input_size:
            raise HTTPException(
                status_code=400,
                detail=f"NLP-translated features length {len(features)} does not match expected input size {expected_input_size}"
            )

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            predicted = torch.argmax(output, dim=1).item()
            confidence = float(probs[predicted]) * 100
            label = threat_labels.get(predicted, "Suspicious or Unknown Behavior")

        # Fallback logic based on confidence and keywords
        lowered = text.lower()
        fallback_class = None

        if confidence < 50:
            if "malware" in lowered or "virus" in lowered:
                fallback_class = 1  # DoS: Back
            elif "sql" in lowered or "injection" in lowered:
                fallback_class = 11  # R2L: Guess Password
            elif "portscan" in lowered or "nmap" in lowered:
                fallback_class = 9  # Probe: Nmap
            elif "phishing" in lowered:
                fallback_class = 11
            elif "hello" in lowered or "hi" in lowered or "weather" in lowered:
                fallback_class = -1  # Safe

        if fallback_class is not None:
            if fallback_class == -1:
                return {
                    "message": "🟢 No significant threat detected.",
                    "class_code": -1,
                    "class_name": "Safe",
                    "confidence": round(confidence, 2)
                }
            else:
                fallback_label = threat_labels.get(fallback_class, "Unknown")
                return {
                    "message": f"⚠️ Threat Detected (Fallback): {fallback_label} (Code: {fallback_class})",
                    "class_code": fallback_class,
                    "class_name": fallback_label,
                    "confidence": round(confidence, 2)
                }

        print(f"✅ NLP Predicted class: {predicted} ({label}) | Confidence: {confidence:.2f}%")

        return {
            "message": f"⚠️ Threat Detected: {label} (Code: {predicted})",
            "class_code": predicted,
            "class_name": label,
            "confidence": round(confidence, 2),
            "features": features
        }

    except Exception as e:
        print("❌ NLP Inference error:", str(e))
        raise HTTPException(status_code=500, detail=f"NLP Model inference failed: {str(e)}")

# ---- Run App ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
