


docker compose down --volumes
rm -rf node_modules package-lock.json
docker compose run app npm install
docker compose build --no-cache
docker compose up 
