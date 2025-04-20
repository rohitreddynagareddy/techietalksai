import Message from '../components/Message';

export default function Home() {
  return (
    <div className="bg-gray-100 p-6 min-h-screen">
      <h1 className="text-3xl font-bold mb-4">Step 2: React fetch from Express backend</h1>
      <p className="mb-4">This page fetches a message from an Express backend running on port 4000.</p>
      <Message />
    </div>
  );
}
