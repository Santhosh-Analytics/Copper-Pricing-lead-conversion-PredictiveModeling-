FROM    python: 3.7
COPY . /app
WORKDIR /app    
RUN pip intall -r requirements.txt
EXPOSE 8501
CMD streamlit  --workers=6 --bind 0.0.0.0:8501 run app.pygit