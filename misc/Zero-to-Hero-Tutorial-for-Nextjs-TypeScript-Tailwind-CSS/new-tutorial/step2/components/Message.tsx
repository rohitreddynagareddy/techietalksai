import React, { useEffect, useState } from 'react';

export default function Message() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    fetch('http://localhost:4000/api/message')
      .then(res => res.json())
      .then(data => setMessage(data.message))
      .catch(() => setMessage('Failed to load message'));
  }, []);

  return <p className="text-lg">{message}</p>;
}
