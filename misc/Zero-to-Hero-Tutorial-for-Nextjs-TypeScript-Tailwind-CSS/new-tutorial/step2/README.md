# Step 2 - React fetch from express backend

This step adds an Express backend with an API endpoint returning a JSON message.

The Next.js frontend React page fetches this message and displays it dynamically.

Structure:
- Express backend running on port 4000 at /api/message
- Next.js frontend page at / which fetches and displays backend message
- React functional component used to fetch and display message
- Tailwind CSS via CDN for styles

To run locally:

1. Run Express backend:
   npx ts-node src/backend/server.ts
2. Run Next.js frontend:
   npm run dev
3. Open http://localhost:3000 to view frontend fetch
4. Backend will be available at http://localhost:4000/api/message