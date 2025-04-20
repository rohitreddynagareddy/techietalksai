import express from 'express';
import cors from 'cors';
const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

app.post('/api/echo', (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }
  res.json({ echo: text });
});

app.listen(port, () => {
  console.log(`Echo server running on port ${port}`);
});