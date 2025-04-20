# Step 3 - Form submit with state and express backend

This step adds form state management and submission in React. The form submits to an Express backend endpoint which echoes back the submitted text as a response.

Structure:
- Express backend running on port 5000 with POST /api/echo endpoint
- React form with input and submit button
- Form submission uses fetch POST and displays response or errors
- Tailwind CSS styles

To run locally:
1. Run backend with: npx ts-node src/backend/server.ts
2. Run frontend with: npm run dev
3. Open http://localhost:3000 to see the form