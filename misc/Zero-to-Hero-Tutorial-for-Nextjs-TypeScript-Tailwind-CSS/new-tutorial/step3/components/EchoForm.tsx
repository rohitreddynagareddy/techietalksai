import React, { useState } from 'react';

export default function EchoForm() {
  const [text, setText] = useState('');
  const [response, setResponse] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResponse('Loading...');
    try {
      const res = await fetch('http://localhost:5000/api/echo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Unknown error');
      }
      setResponse(data.echo);
    } catch (err: any) {
      setError(err.message);
      setResponse('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col space-y-4 max-w-md">
      <input
        type="text"
        placeholder="Enter text to echo"
        className="border border-gray-400 p-2 rounded"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button type="submit" className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Submit</button>
      {response && <p className="text-green-700 font-semibold">{response}</p>}
      {error && <p className="text-red-700 font-semibold">{error}</p>}
    </form>
  );
}
