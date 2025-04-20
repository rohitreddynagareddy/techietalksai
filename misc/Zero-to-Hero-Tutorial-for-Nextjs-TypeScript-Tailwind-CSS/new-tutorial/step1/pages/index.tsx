import React from 'react';
import Head from 'next/head';

export default function Home() {
  return (
    <div className="bg-gray-100 p-6">
      <Head>
        <title>Step 1 - Basic Next.js API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <script src="https://cdn.tailwindcss.com"></script>
      </Head>
      <h1 className="text-3xl font-bold mb-4">Step 1: Basic Next.js + API</h1>
      <p className="mb-2">This is a simple Next.js page with an API route.</p>
      <p className="italic">Open <code>/api/hello</code> to see the API response JSON.</p>
    </div>
  );
}