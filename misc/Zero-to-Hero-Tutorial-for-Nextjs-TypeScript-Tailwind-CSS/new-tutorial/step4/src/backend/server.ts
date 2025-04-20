import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';

const app = express();
const port = 6000;

app.use(cors());
app.use(express.json());

const mongoUrl = process.env.MONGO_URL || 'mongodb://mongo:27017/notesdb';
mongoose.connect(mongoUrl).then(() => {
  console.log('Connected to MongoDB');
}).catch(err => {
  console.error('MongoDB connection error:', err);
});

const noteSchema = new mongoose.Schema({
  text: String,
});
const Note = mongoose.model('Note', noteSchema);

app.post('/api/notes', async (req, res) => {
  try {
    const note = new Note({ text: req.body.text });
    await note.save();
    res.json(note);
  } catch (err) {
    res.status(500).json({ error: 'Failed to save note' });
  }
});

app.get('/api/notes', async (req, res) => {
  try {
    const notes = await Note.find();
    res.json(notes);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch notes' });
  }
});

app.listen(port, () => {
  console.log(`Notes server running on port ${port}`);
});