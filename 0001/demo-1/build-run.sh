

docker build -t tmp .
docker run -v $PWD:/app tmp $1