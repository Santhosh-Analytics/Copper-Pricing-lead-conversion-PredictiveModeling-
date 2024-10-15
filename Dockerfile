FROM python:3.7
COPY . /app
WORKDIR /app    
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 8501
CMD [ "streamlit","run","app.py" ]