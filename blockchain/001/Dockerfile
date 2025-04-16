FROM python:3.11-slim
WORKDIR /app
COPY blockchain/ /app/
RUN pip install flask
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["main.py"]
