FROM dregistry.dvl.mc/analytics/ceo-classifier:base

USER app
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt \
    && python3 -c "import nltk; nltk.download('stopwords')" \
    && echo "Requirements were installed."

EXPOSE 8083

CMD ["python", "service.py"]
