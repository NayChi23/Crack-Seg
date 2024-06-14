
from flask import Flask, render_template, url_for, redirect, request, send_file, session
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
import base64
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
# Additional imports for new functionality

import json
import torch
import torchaudio
import io
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import os


import cv2
from PIL import Image


import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from mat_extract import descriptor_mat
from torch_geometric.data import Data
from extractor import ViTExtractor
from gnn_pool import GNNpool
import torch.optim as optim
from tqdm import tqdm
import util



from matplotlib import cm
from numba import njit
import urllib.request
import warnings
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

db = SQLAlchemy(app)
migrate = Migrate(app,db)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    score = db.Column(db.Integer, default=0)
    
class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    level = SelectField('Select Level', choices=[('level1', 'Level 1'), ('level2', 'Level 2')], validators=[InputRequired()])

    submit = SubmitField('Login')

def segment_image(file_stream):
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    K = 2
    epoch = 10
    res = (224, 224)
    stride = 4
    facet = 'key'
    layer = 11
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_bin = False
    cc = False

    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)
    uploaded_image = Image.open(file_stream).convert('RGB')
    prep = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image_tensor = prep(uploaded_image)[None, ...]
    image_np = np.array(uploaded_image)
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    feats_dim = 384
    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()
    W, F, D = descriptor_mat(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
    node_feats, edge_index, edge_weight = util.load_data(W, F)
    data = Data(node_feats, edge_index, edge_weight).to(device)
    model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
    opt = optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(epoch):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()
    S = S.detach().cpu()
    S = torch.argmax(S, dim=-1)
    mask0, S = util.graph_to_mask(S, cc, stride, image_tensor, image_np)
    mask0_image = Image.fromarray(mask0 * 255).convert('L')
    # Convert the segmented mask to a numpy array
    mask0_np = np.array(mask0_image)

    # Convert the original image and mask to the same data type
    image_np = np.array(uploaded_image).astype(np.uint8)
    # Convert the segmented mask to a numpy array
    mask0_np = mask0_np.astype(np.uint8)

    # Create a color mask
    color_mask = np.zeros_like(image_np)
    segmented_areas = cv2.applyColorMap(mask0_np * 255, cv2.COLORMAP_JET)
    color_mask[mask0_np > 0] = segmented_areas[mask0_np > 0]

    # Overlay the color mask on the image
    overlay = cv2.addWeighted(image_np, 1.0, color_mask, 0.7, 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask0_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the segmented areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Convert the overlay image to a PIL Image
    overlay_image = Image.fromarray(overlay)

    return overlay_image


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                # Redirect to different dashboard pages based on the selected level
                if form.level.data == "level1":
                    return redirect(url_for('dashboard'))
                elif form.level.data == "level2":
                    return redirect(url_for('dashboard_level2'))
    return render_template('login.html', form=form)


# @app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
# def dashboard():
#     if request.method == 'POST':
#         file_stream = request.files['file']
#         segmented_image = segment_image(file_stream)

#         # Convert PIL Image to bytes
#         img_io = BytesIO()
#         segmented_image.save(img_io, 'PNG')
#         img_io.seek(0)
#         base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')

#         # Pass the encoded image to template
#         return render_template('dashboard.html', segmented_image_data=base64_img)

#     return render_template('dashboard.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        file_stream = request.files['file']
        segmented_image = segment_image(file_stream)

        # Convert PIL Image to bytes
        img_io = BytesIO()
        segmented_image.save(img_io, 'PNG')
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # Update the user's score
        current_user.score += 1  # Increment score by 10 points or any other logic
        db.session.commit()  # Commit changes to the database

        # Pass the encoded image and updated score to the template
        return render_template('dashboard.html', segmented_image_data=base64_img, score=current_user.score)

    # Initially or on GET request, display the current score if it exists
    return render_template('dashboard.html', score=current_user.score)

@app.route('/redeem')
@login_required  # Ensure only logged-in users can access this route
def redeem_points():
    return render_template('redeem.html')

@app.route('/output_image')
@login_required  # Ensure only logged-in users can access this route
def output_image():
    file_path = r"C:\Users\ASUS\Downloads\EIE463\EIE463\static\output.png"

    return send_file(file_path, mimetype='image/png')

@app.route('/segmented_image')
@login_required
def segmented_image_page():
    # base64_img = session.get('segmented_image_data')  # Get from session, not db.session
    # if not base64_img:
    #     return redirect(url_for('dashboard_level2'))  # Redirect if no image data

    return render_template('segmented_image.html')


@app.route('/dashboard_level2', methods=['GET', 'POST'])
@login_required
def dashboard_level2():
    if request.method == 'POST':
        file_stream = request.files['file']
        segmented_image = segment_image(file_stream)

        # Convert PIL Image to bytes
        img_io = BytesIO()
        segmented_image.save(img_io, 'PNG')
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Store in session
        session['segmented_image_data'] = base64_img

        db.session.commit()  # If there's anything to commit to the database

    return render_template('dashboard_level2.html', segmented_image_data=session.get('segmented_image_data'))


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
