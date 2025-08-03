import tensorflow as tf
import gradio as gr
import numpy as np

# This file (cxr_pneumonia_model.keras) must be present in your repo!
model = tf.keras.models.load_model('cxr_pneumonia_model.keras')

def predict_xray(img):
    img = img.resize((224,224))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    return "PNEUMONIA" if pred > 0.5 else "NORMAL"

iface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type='pil'),
    outputs=gr.Text(),
    title='Chest X-ray Pneumonia Detection',
    description='Upload a chest X-ray image to predict NORMAL or PNEUMONIA.'
)

if __name__ == "__main__":
    iface.launch()
