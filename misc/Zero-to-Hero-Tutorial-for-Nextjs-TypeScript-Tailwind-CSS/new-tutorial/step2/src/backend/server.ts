import express from 'express';
import cors from 'cors';
const app = express();
const port = 4000;

app.use(cors());
app.use(express.json());

app.get('/api/message', (req, res) => {
  res.json({ message: 'Hello from Express backend!' });
});

app.listen(port, () => {
  console.log(`Express backend server running on port ${port}`);
});