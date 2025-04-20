import EchoForm from '../components/EchoForm';

export default function Home() {
  return (
    <div className="bg-gray-100 p-6 min-h-screen">
      <h1 className="text-3xl font-bold mb-4">Step 3: Form with state and express backend submit</h1>
      <p className="mb-4">Enter text in the form below; it will be submitted to the backend and echoed back.</p>
      <EchoForm />
    </div>
  );
}
