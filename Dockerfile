FROM python:3.6-slim
LABEL maintainer="Fakhruddin Shaukat <fakhruddin.shaukat@hotmail.com>"

EXPOSE 8080

RUN apt-get update && apt-get install -y build-essential python-dev \
&& pip3 install --no-cache-dir "pillow==5.2.0" \
&& pip3 install --no-cache-dir "tensorflow==1.10.0" \
&& pip3 install --no-cache-dir "keras==2.2.2" \
&& pip3 install --no-cache-dir "flask==1.0.2" \
&& pip3 install --no-cache-dir "flask-cors==3.0.6" \
&& pip3 install --no-cache-dir "uwsgi==2.0.17.1" 

WORKDIR /app

COPY ./serve.py ./
COPY ./models ./models

ENTRYPOINT ["uwsgi", "--http", ":8080", "--wsgi-file","serve.py","--callable","app"]