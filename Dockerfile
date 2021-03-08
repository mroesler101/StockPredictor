FROM python:3

WORKDIR StockPredictor/
COPY . /StockPredictor

EXPOSE 8080

RUN pip install -r requirements.txt

CMD ["make", "start"]
