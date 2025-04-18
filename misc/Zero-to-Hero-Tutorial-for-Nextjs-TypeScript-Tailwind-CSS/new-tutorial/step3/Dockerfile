FROM node:18\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 3000 5000\nCMD [ "bash", "-c", "npm run backend & npm run dev" ]
