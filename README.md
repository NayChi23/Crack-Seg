# CrackSense

## Overview

CrackSense is a machine learning-driven web application designed to detect and analyze structural damage, specifically cracks, in various infrastructures. Utilizing state-of-the-art image processing algorithms, CrackSense provides an easy-to-use interface for engineers and safety inspectors to upload images of infrastructures, receive instant assessments, and manage historical analysis data.

## Features

- **Image Upload**: Users can upload images directly through the web interface.
- **Real-time Processing**: The application processes images in real-time to detect structural damage.
- **Dashboard**: Interactive dashboard to view and analyze past results and trends.
- **Security**: Secure login and user management system.
- **Mobile Friendly**: Responsive design that works on both desktop and mobile browsers.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip and virtualenv
- A modern web browser

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/CrackSense.git
   cd CrackSense
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   virtualenv venv
   # Windows
   .\venv\Scripts\activate
   # Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database**
   ```bash
   flask db upgrade
   ```

5. **Run the Application**
   ```bash
   flask run
   ```

   Access the application through `http://localhost:5000` in your web browser.

## Usage

1. **Register/Login**: Access the web application through your browser and register or log in.
2. **Uploading Images**: Navigate to the dashboard and upload images using the provided form.
3. **View Results**: After processing, the results will be displayed directly on the dashboard.

## Configuration

- **Database Configuration**: Modify the `app.config['SQLALCHEMY_DATABASE_URI']` in `app.py` to connect to your preferred database.
- **Port Configuration**: By default, the application runs on port 5000. This can be changed in the `app.run()` method in `app.py`.

## Contributing

We welcome contributions to the CrackSense project. Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
