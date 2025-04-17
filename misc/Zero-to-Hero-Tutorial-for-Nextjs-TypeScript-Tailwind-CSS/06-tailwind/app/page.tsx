"use client";

import { useState } from "react";

export default function TailwindDemoPage() {
  const [count, setCount] = useState(0);
  const [darkMode, setDarkMode] = useState(false);

  return (
    <main className={`${darkMode ? "dark" : ""} min-h-screen bg-gray-100 dark:bg-gray-900 p-6`}>      
      <div className="max-w-3xl mx-auto space-y-6">
        <h1 className="text-4xl font-bold text-center text-blue-600 dark:text-blue-300">
          Tailwind CSS Live Demo
        </h1>

        <div className="flex justify-between items-center p-4 border rounded-lg shadow bg-white dark:bg-gray-800">
          <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
            Count: <span className="font-bold text-indigo-600 dark:text-indigo-300">{count}</span>
          </p>
          <div className="flex space-x-2">
            <button
              onClick={() => setCount(count + 1)}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition"
            >
              +1
            </button>
            <button
              onClick={() => setCount(0)}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition"
            >
              Reset
            </button>
          </div>
        </div>

        <section className="grid md:grid-cols-2 gap-4">
          <div className="p-4 border rounded bg-white dark:bg-gray-800">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2">Layout & Spacing</h2>
            <div className="flex flex-wrap gap-2">
              <div className="w-12 h-12 bg-blue-400"></div>
              <div className="w-16 h-16 bg-green-400"></div>
              <div className="w-20 h-20 bg-red-400"></div>
            </div>
          </div>

          <div className="p-4 border rounded bg-white dark:bg-gray-800">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2">Typography</h2>
            <p className="text-sm text-gray-500">This is small gray text.</p>
            <p className="text-base text-black dark:text-white font-medium">Normal base text.</p>
            <p className="text-lg font-bold text-purple-600">Large bold text.</p>
          </div>
        </section>

        <div className="text-center">
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="mt-4 px-6 py-2 border border-gray-400 dark:border-gray-600 rounded-full text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
          >
            Toggle {darkMode ? "Light" : "Dark"} Mode
          </button>
        </div>

        <footer className="text-center text-xs text-gray-400 pt-8">
          Made with Tailwind CSS
        </footer>
      </div>
    </main>
  );
}
