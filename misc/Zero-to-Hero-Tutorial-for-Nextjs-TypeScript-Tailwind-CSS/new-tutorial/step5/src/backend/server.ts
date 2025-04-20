import express from 'express';
import cors from 'cors';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const app = express();
const port = 7000;
const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret';

app.use(cors());
app.use(express.json());

const user = { 
  id: 1, 
  username: 'user', 
  passwordHash: bcrypt.hashSync('pass', 8) 
};

app.post('/api/login', (req, res) => {
  const { username, password } = req.body;
  if (username !== user.username || !bcrypt.compareSync(password, user.passwordHash)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '1h' });
  res.json({ token });
});

function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Missing token' });
  
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: 'Invalid token' });
    req.user = user;
    next();
  });
}

app.get('/api/protected', authenticateToken, (req, res) => {
  const userData = req.user || {}
  res.json({ message: `Hello, ${userData.username}! This is protected data.` });
});

app.listen(port, () => {
  console.log('Auth backend listening on port ' + port);
});