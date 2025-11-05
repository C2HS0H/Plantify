import io 
import torch
import base64
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request
import torchvision.transforms.functional as TF

import CNN
 
MODEL_PATH = "model.pt"
DISEASE_CSV = "disease_info.csv"
SUPPLEMENT_CSV = "supplement_info.csv"

disease_info = pd.read_csv(DISEASE_CSV, encoding='utf8')
supplement_info = pd.read_csv(SUPPLEMENT_CSV, encoding='utf8')

model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def prediction(image):
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def submit():
    if request.method == 'POST':
        image_b64 = request.form.get('image_b64')
        if not image_b64:
            return "No image data received", 400

        image_data = image_b64.split(',')[1]
        decoded_image = base64.b64decode(image_data)
        image_stream = io.BytesIO(decoded_image)
        image = Image.open(image_stream)

        pred = prediction(image)
        
        healthy_pred_ids = [3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38]
        full_disease_name_from_csv = disease_info['disease_name'][pred]

        standardized_name = full_disease_name_from_csv.replace(" : ", "|").replace("___", "|")
        parts = standardized_name.split('|')
        plant_name = parts[0].replace("_", " ").strip()

        if pred in healthy_pred_ids:
            disease_name = "Healthy"
            display_diagnosis = f"The {plant_name.lower()} leaves in the image appear healthy and thriving!"
        else:
            disease_only = parts[1].replace("_", " ").strip()
            display_diagnosis = f"The {plant_name.lower()} leaves in the image show signs of {disease_only.lower()}."
            disease_name = f"{plant_name} {disease_only}"

        disease_description = disease_info['disease_description'][pred]
        prevent = disease_info['recommended_actions'][pred]
        image_url = disease_info['image_url'][pred]
        recommended_product = supplement_info['recommended_product'][pred]
        product_image_url_url = supplement_info['product_image_url'][pred]
        supplement_purchase_url = supplement_info['purchase_url'][pred]
        
        return render_template('diagnose.html', 
                               pred=pred,
                               healthy_pred_ids=healthy_pred_ids,
                               plant_name=plant_name,
                               disease_name=disease_name,
                               display_diagnosis=display_diagnosis,
                               desc=disease_description, 
                               prevent=prevent, 
                               image_url=image_url, 
                               sname=recommended_product, 
                               simage=product_image_url_url, 
                               purchase_url=supplement_purchase_url,
                               uploaded_image_b64=image_b64)
    
@app.route('/supplements')
def market():
    products = []
    for idx, row in supplement_info.iterrows():
        if pd.notna(row['recommended_product']) and row['recommended_product'].strip():
            
            card_type = 'healthy' if 'healthy' in row['disease_name'] else 'diseased'
            
            product = {
                'name': row['recommended_product'],
                'image': row['product_image_url'],
                'purchase_url': row['purchase_url'],
                'type': card_type
            }
            products.append(product)
            
    return render_template('supplements.html', products=products)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
