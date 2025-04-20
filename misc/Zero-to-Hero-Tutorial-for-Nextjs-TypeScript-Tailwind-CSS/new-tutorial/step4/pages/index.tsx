import Notes from '../components/Notes';

export default function Home() {
  return (
    <div className="bg-gray-100 p-6 min-h-screen">
      <h1 className="text-3xl font-bold mb-4">Step 4: MongoDB CRUD in Express backend + React frontend</h1>
      <p className="mb-4">Add notes to MongoDB through Express backend and list notes in React.</p>
      <Notes />
    </div>
  );
}
