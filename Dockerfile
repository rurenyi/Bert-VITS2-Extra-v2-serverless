FROM registry.cn-hangzhou.aliyuncs.com/serverless_devs/pytorch:22.12-py3

WORKDIR /app

COPY requirements.txt /app
VOLUME ["/app/data","/app/emotional","/app/g2pW","/app/bert","/app/slm"]

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-deps vector_quantize_pytorch

COPY . /app

EXPOSE 5000

CMD ["python", "serverless.py"]