pip install -r requirements.txt
python train_models.py
gunicorn --bind=0.0.0.0 --timeout 600 app:app
