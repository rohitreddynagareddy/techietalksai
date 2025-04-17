import Link from 'next/link'

export default function Home() {
  return (
    <div className="container">
      <h1>Welcome to Sudoku Game</h1>
      <p><Link href="/sudoku"><a>Play Sudoku</a></Link></p>
    </div>
  )
}
