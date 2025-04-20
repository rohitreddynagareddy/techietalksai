import React, { useState } from 'react';

export default function LoginForm() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState('');
  const [protectedData, setProtectedData] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    try {
      const res = await fetch('http://localhost:7000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Login failed');
      }
      
      setToken(data.token);
      setPassword(''); // Clear password for security
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchProtectedData = async () => {
    if (!token) {
      setError('No token available. Please login first.');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      const res = await fetch('http://localhost:7000/api/protected', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Failed to fetch protected data');
      }
      
      setProtectedData(data.message);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      {!token ? (
        <form onSubmit={handleLogin} className="flex flex-col space-y-4 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Login</h2>
          
          <div>
            <label className="block text-gray-700 mb-1">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full border border-gray-300 p-2 rounded"
              required
            />
          </div>
          
          <div>
            <label className="block text-gray-700 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border border-gray-300 p-2 rounded"
              required
            />
          </div>
          
          <button 
            type="submit" 
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
            disabled={isLoading}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
          
          {error && <p className="text-red-600 mt-2">{error}</p>}
          <p className="text-sm text-gray-600 mt-2">Use username: "user" and password: "pass"</p>
        </form>
      ) : (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Logged In Successfully</h2>
          <p className="mb-4">You have a valid JWT token.</p>
          
          {protectedData ? (
            <div className="p-4 bg-green-100 border border-green-200 rounded-md mb-4">
              <h3 className="font-medium text-green-800 mb-1">Protected Data:</h3>
              <p className="text-green-700">{protectedData}</p>
            </div>
          ) : (
            <button
              onClick={fetchProtectedData}
              className="bg-indigo-600 text-white py-2 px-4 rounded hover:bg-indigo-700 disabled:opacity-50 mb-4"
              disabled={isLoading}
            >
              {isLoading ? 'Loading...' : 'Fetch Protected Data'}
            </button>
          )}
          
          <button
            onClick={() => {
              setToken('');
              setProtectedData('');
              setError('');
            }}
            className="bg-gray-500 text-white py-2 px-4 rounded hover:bg-gray-600"
          >
            Logout
          </button>
          
          {error && <p className="text-red-600 mt-2">{error}</p>}
        </div>
      )}
    </div>
  );
}
