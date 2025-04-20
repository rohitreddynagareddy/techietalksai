import React, { useState, useEffect } from 'react';

export default function Notes() {
  const [text, setText] = useState('');
  const [notes, setNotes] = useState<Array<{ _id: string; text: string }>>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('http://localhost:6000/api/notes')
      .then(res => res.json())
      .then(setNotes)
      .catch(() => setError('Failed to load notes'));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      const res = await fetch('http://localhost:6000/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error('Submit failed');
      const note = await res.json();
      setNotes([...notes, note]);
      setText('');
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
        <input
          type="text"
          placeholder="New note"
          className="border border-gray-400 p-2 rounded"
          value={text}
          onChange={e => setText(e.target.value)}
          required
        />
        <button type="submit" className="bg-indigo-600 hover:bg-indigo-800 text-white font-bold py-2 px-4 rounded">Add Note</button>
      </form>
      {error && <p className="text-red-600 mt-2">{error}</p>}
      <ul className="mt-4 list-disc list-inside">
        {notes.map(note => (
          <li key={note._id}>{note.text}</li>
        ))}
      </ul>
    </div>
  );
}
