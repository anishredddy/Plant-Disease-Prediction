import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from flask import Flask
from flask_cors import CORS



disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index



app = Flask(__name__)
cors = CORS(app, resources={r'/api/*' : {'origins':'*'}}) 
def api_home_page():
    data = {
        "message": "Welcome to the home page of your app!",
        # Add more data as needed
    }
    return jsonify(data)

@app.route('/api/contact')
def api_contact():
    data = {
        "message": "Contact information for your app.",
        # Add more data as needed
    }
    return jsonify(data)

@app.route('/api/index')
def api_ai_engine_page():
    data = {
        "message": "AI engine page data for your app.",
        # Add more data as needed
    }
    return jsonify(data)

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/api/submit', methods=['POST'])
def api_submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        data = {
            "title": title,
            "description": description,
            "prevent": prevent,
            "image_url": image_url,
            "pred": pred,
            "supplement_name": supplement_name,
            "simage": supplement_image_url,
            "buy_link": supplement_buy_link
        }
        return jsonify(data)

@app.route('/api/market')
def api_market():
    data = {
        "supplement_image": list(supplement_info['supplement image']),
        "supplement_name": list(supplement_info['supplement name']),
        "disease": list(disease_info['disease_name']),
        "buy": list(supplement_info['buy link'])
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
