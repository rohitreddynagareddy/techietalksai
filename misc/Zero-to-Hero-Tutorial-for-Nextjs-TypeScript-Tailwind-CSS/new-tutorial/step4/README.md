# Step 4 - MongoDB CRUD with Express + React

This step demonstrates connecting an Express backend to MongoDB (using mongoose) to store and retrieve notes. The React frontend provides a form to add notes and lists all notes from the database.

You must have MongoDB running. The included docker-compose.yml spins up a mongo service at mongodb://mongo:27017/notesdb.

To run locally:

1. Start MongoDB (docker run or docker-compose up mongo)
2. Start backend with: npx ts-node src/backend/server.ts
3. Start frontend with: npm run dev
4. Open http://localhost:3000 to view app