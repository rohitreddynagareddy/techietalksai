FROM node:18
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000 4000

CMD [ "bash", "-c", "npm run backend & npm run dev" ]