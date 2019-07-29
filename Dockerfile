FROM python:3.7.2

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install gunicorn -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

COPY ./ /app/

CMD ["gunicorn","-w","4","-t","1000","-b","0.0.0.0:80","app:app"]