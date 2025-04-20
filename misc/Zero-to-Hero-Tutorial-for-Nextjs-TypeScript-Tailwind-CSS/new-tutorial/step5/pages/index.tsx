import React from 'react';
import LoginForm from '../components/LoginForm';

export default function Home() {
  return (
    <div className="bg-gray-100 p-6 min-h-screen">
      <h1 className="text-3xl font-bold mb-4">Step 5: JWT Authentication with Express + React</h1>
      <p className="mb-6">This page demonstrates a simple login process using JWT tokens for authentication.</p>
      <LoginForm />
    </div>
  );
}
