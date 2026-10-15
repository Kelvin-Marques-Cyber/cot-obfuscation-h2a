FROM python:3.12.14-slim
WORKDIR /work
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
CMD ["bash", "run.sh"]
