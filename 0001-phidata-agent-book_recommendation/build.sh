

docker build -t phidata-agent-001 .
docker run -p 4000:80 --env-file .env phidata-agent-001 book_recommendation.py 
