"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function NextJSDemoPage() {
  const router = useRouter();
  const [name, setName] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert(`Hello, ${name}! Navigating to /greet...`);
    router.push("/greet");
  };

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold text-blue-700 text-center">
          Next.js Features Demo
        </h1>

        {/* Client-side Navigation */}
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold mb-2">Client-side Navigation</h2>
          <p className="mb-2 text-gray-700">
            Go to a different route using <code>Link</code>:
          </p>
          <Link
            href="/about"
            className="text-indigo-600 hover:underline"
          >
            Go to About Page
          </Link>
        </div>

        {/* useRouter for programmatic navigation */}
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold mb-2">useRouter</h2>
          <form onSubmit={handleSubmit} className="space-y-2">
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="border p-2 rounded w-full"
              placeholder="Enter your name"
            />
            <button
              type="submit"
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Greet Me
            </button>
          </form>
        </div>

        {/* Image optimization example */}
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold mb-2">Image Optimization</h2>
          <p className="mb-2 text-gray-700">An optimized image using Next.js:</p>
          <img
            src="https://placekitten.com/400/300"
            alt="A kitten"
            width={400}
            height={300}
            className="rounded border"
          />
        </div>

        {/* API route demonstration */}
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold mb-2">API Route Fetch</h2>
          <button
            onClick={async () => {
              const res = await fetch("/api/hello");
              const data = await res.json();
              alert(`API says: ${data.message}`);
            }}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          >
            Call /api/hello
          </button>
        </div>
      </div>
    </main>
  );
}
