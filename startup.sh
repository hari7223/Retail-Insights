pip install -r requirements.txt
gunicorn --chdir /home/site/wwwroot --bind=0.0.0.0 --timeout 600 --workers 1 --worker-class gthread --threads 4 app:app
