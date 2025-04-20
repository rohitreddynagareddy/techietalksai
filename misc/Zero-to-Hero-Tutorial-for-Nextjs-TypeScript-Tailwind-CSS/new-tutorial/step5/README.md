# Step 5 - JWT Authentication with Express + React

This step demonstrates a minimal JWT authentication backend with express, and a React login form that obtains a JWT token and fetches protected data using the token.

- Login endpoint: POST /api/login with username and password
- Protected data endpoint: GET /api/protected, requires Bearer JWT token
- React login form stores token and fetches protected data
- Hardcoded user credentials: user / pass

To run locally:
1. Run backend with: npx ts-node src/backend/server.ts
2. Run frontend with: npm run dev
3. Open http://localhost:3000
4. Login and fetch protected data