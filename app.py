import gradio as gr
import requests

def predict_sentiment(text):
    response = requests.post(
        "http://localhost:8000/predict/",
        json={"text": text}
    )
    if response.status_code == 200:
        result = response.json()
        return f"{result['sentiment']} (Confidence: {result['confidence']:.2f})"
    return "Error predicting sentiment"

interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter your text here..."),
    outputs=gr.Textbox(label="Sentiment Prediction"),
    title="Sentiment Analysis",
    description="Enter text to analyze its sentiment (Positive/Negative)",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(server_port=7860, share=False)