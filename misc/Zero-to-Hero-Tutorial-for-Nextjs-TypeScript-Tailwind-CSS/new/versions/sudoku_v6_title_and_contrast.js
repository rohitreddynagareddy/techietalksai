import React, { useState, useEffect, useRef } from 'react'

function deepCopy(grid) {
  return grid.map(row => row.slice())
}

function isValid(grid, row, col, num) {
  for (let c = 0; c < 9; c++) {
    if (grid[row][c] === num) return false
  }
  for (let r = 0; r < 9; r++) {
    if (grid[r][col] === num) return false
  }
  const boxRow = Math.floor(row / 3) * 3
  const boxCol = Math.floor(col / 3) * 3
  for (let r = boxRow; r < boxRow + 3; r++) {
    for (let c = boxCol; c < boxCol + 3; c++) {
      if (grid[r][c] === num) return false
    }
  }
  return true
}

function solve(board) {
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      if (board[row][col] === 0) {
        const nums = shuffleArray([1, 2, 3, 4, 5, 6, 7, 8, 9])
        for (const num of nums) {
          if (isValid(board, row, col, num)) {
            board[row][col] = num
            if (solve(board)) {
              return true
            }
            board[row][col] = 0
          }
        }
        return false
      }
    }
  }
  return true
}

function shuffleArray(arr) {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function generateFullBoard() {
  const board = Array(9)
    .fill(null)
    .map(() => Array(9).fill(0))
  solve(board)
  return board
}

function generatePuzzle(fullBoard, holes = 40) {
  const puzzle = deepCopy(fullBoard)
  let removed = 0
  while (removed < holes) {
    const r = Math.floor(Math.random() * 9)
    const c = Math.floor(Math.random() * 9)
    if (puzzle[r][c] !== 0) {
      puzzle[r][c] = 0
      removed++
    }
  }
  return puzzle
}

const TIMER_BY_LEVEL = {
  easy: 300,
  medium: 600,
  hard: 900,
}

export default function Sudoku() {
  const [startingPuzzle, setStartingPuzzle] = useState(null)
  const [grid, setGrid] = useState(null)
  const [solvedBoard, setSolvedBoard] = useState(null)
  const [selected, setSelected] = useState(null)
  const [errorCells, setErrorCells] = useState([])
  const [revealedCells, setRevealedCells] = useState([])
  const [difficulty, setDifficulty] = useState('easy')
  const [timeLeft, setTimeLeft] = useState(TIMER_BY_LEVEL['easy'])

  const timerId = React.useRef(null)

  function generateNewPuzzle(level) {
    const fullBoard = generateFullBoard()
    let holes = 45
    if (level === 'easy') holes = 35
    else if (level === 'hard') holes = 55
    const puzzle = generatePuzzle(fullBoard, holes)
    setStartingPuzzle(puzzle)
    setGrid(deepCopy(puzzle))
    setSolvedBoard(fullBoard)
    setSelected(null)
    setErrorCells([])
    setRevealedCells([])
    setTimeLeft(TIMER_BY_LEVEL[level])
  }

  React.useEffect(() => {
    generateNewPuzzle(difficulty)
  }, [difficulty])

  React.useEffect(() => {
    if (timeLeft <= 0) {
      setSelected(null)
      if (timerId.current) {
        clearInterval(timerId.current)
        timerId.current = null
      }
      return
    }
    if (!timerId.current) {
      timerId.current = setInterval(() => {
        setTimeLeft((t) => (t > 0 ? t - 1 : 0))
      }, 1000)
    }
    return () => {
      if (timerId.current) {
        clearInterval(timerId.current)
        timerId.current = null
      }
    }
  }, [timeLeft])

  function handleCellClick(row, col) {
    if (!startingPuzzle || startingPuzzle[row][col] !== 0 || timeLeft <= 0) {
      setSelected(null)
      return
    }
    setSelected([row, col])
    setErrorCells([])
  }

  function handleNumberClick(num) {
    if (!selected || !grid || timeLeft <= 0) return
    const [r, c] = selected
    setGrid((oldGrid) => {
      const newGrid = deepCopy(oldGrid)
      newGrid[r][c] = num
      return newGrid
    })
    if (num === 0) {
      setRevealedCells((old) => old.filter(([rr, cc]) => rr !== r || cc !== c))
    } else {
      setRevealedCells((old) => old.filter(([rr, cc]) => rr !== r || cc !== c))
    }
    setSelected(null)
  }

  function handleRevealClick() {
    if (!selected || !grid || !solvedBoard || timeLeft <= 0) return
    const [r, c] = selected
    setGrid((oldGrid) => {
      const newGrid = deepCopy(oldGrid)
      newGrid[r][c] = solvedBoard[r][c]
      return newGrid
    })
    setRevealedCells((old) => {
      const exists = old.some(([rr, cc]) => rr === r && cc === c)
      if (exists) return old
      return [...old, [r, c]]
    })
    setSelected(null)
  }

  React.useEffect(() => {
    if (!grid) return
    let errs = []
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const val = grid[r][c]
        if (val !== 0) {
          grid[r][c] = 0
          if (!isValid(grid, r, c, val)) {
            errs.push([r, c])
          }
          grid[r][c] = val
        }
      }
    }
    setErrorCells(errs)
  }, [grid])

  function isErrorCell(r, c) {
    return errorCells.some(([rr, cc]) => rr === r && cc === c)
  }

  function isRevealedCell(r, c) {
    return revealedCells.some(([rr, cc]) => rr === r && cc === c)
  }

  function blockColor(row, col) {
    const blockId = Math.floor(row / 3) + Math.floor(col / 3) * 3
    const colors = ['#c9e4ff', '#ffc9de', '#fff1c9']
    return colors[blockId % colors.length]
  }

  const formatTime = (seconds) => {
    const mm = Math.floor(seconds / 60).toString().padStart(2, '0')
    const ss = (seconds % 60).toString().padStart(2, '0')
    return `${mm}:${ss}`
  }

  const timerPercent = (timeLeft / TIMER_BY_LEVEL[difficulty]) * 100

  return (
    <div className="sudoku-page">
      <div className="timer-bar-container">
        <div
          className="timer-bar"
          style={{ width: `${timerPercent}%` }}
          aria-label={`Time left: ${formatTime(timeLeft)}`}
        ></div>
        <span className="timer-text">{formatTime(timeLeft)}</span>
      </div>

      <div className="difficulty-selector">
        <label htmlFor="difficulty">Difficulty:</label>
        <select
          id="difficulty"
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value)}
          disabled={timeLeft <= 0}
        >
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>

      <h1>Sree's Sudoku Advanced</h1>

      <div className="number-pad">
        {Array.from({ length: 9 }, (_, i) => (
          <button
            key={i + 1}
            onClick={() => handleNumberClick(i + 1)}
            className="number-button"
            disabled={!selected || timeLeft <= 0}
          >
            {i + 1}
          </button>
        ))}
        <button
          className="number-button clear"
          onClick={() => handleNumberClick(0)}
          disabled={!selected || timeLeft <= 0}
        >
          Clear
        </button>
        <button
          className="number-button reveal"
          onClick={handleRevealClick}
          disabled={!selected || timeLeft <= 0}
        >
          Reveal
        </button>
      </div>

      {!startingPuzzle || !grid ? (
        <div>Loading...</div>
      ) : (
        <div className="sudoku-grid">
          {grid.map((row, r) => (
            <div key={r} className="sudoku-row">
              {row.map((cell, c) => {
                const isStarting = startingPuzzle[r][c] !== 0
                const isSelected = selected && selected[0] === r && selected[1] === c
                const cellError = isErrorCell(r, c)
                const isRevealed = isRevealedCell(r, c)
                return (
                  <div
                    key={c}
                    className={`sudoku-cell ${
                      isStarting ? 'starting-cell' : ''
                    } ${isSelected ? 'selected-cell' : ''} ${
                      cellError ? 'error-cell' : ''
                    }`}
                    onClick={() => handleCellClick(r, c)}
                    style={{ backgroundColor: blockColor(r, c) }}
                  >
                    {cell !== 0 ? (
                      <span
                        className={`${
                          isStarting
                            ? 'starting-number'
                            : isRevealed
                            ? 'revealed-number'
                            : 'user-number'
                        }`}
                      >
                        {cell}
                      </span>
                    ) : (
                      ''
                    )}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      )}

      <style jsx>{`
        .timer-bar-container {
          position: relative;
          height: 20px;
          width: 100%;
          max-width: 520px;
          margin: 0 auto 15px auto;
          background: #ddd;
          border-radius: 10px;
          overflow: hidden;
          box-shadow: inset 0 0 5px #aaa;
        }

        .timer-bar {
          height: 100%;
          background: linear-gradient(90deg, #ff6b6b, #f06595, #ff6b6b);
          transition: width 1s linear;
        }

        .timer-text {
          position: absolute;
          top: 0;
          left: 50%;
          transform: translateX(-50%);
          font-weight: 700;
          color: #333;
          text-shadow: 0 0 2px #fff;
          user-select: none;
        }

        .difficulty-selector {
          margin: 0 auto 10px auto;
          max-width: 200px;
          font-size: 18px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          color: #222;
          font-weight: 600;
        }

        .difficulty-selector select {
          padding: 5px 8px;
          font-size: 16px;
          border-radius: 6px;
          border: 1px solid #444;
          cursor: pointer;
          user-select: none;
          transition: border-color 0.3s ease;
        }

        .difficulty-selector select:hover:enabled {
          border-color: #667eea;
        }

        .difficulty-selector select:disabled {
          background: #eee;
          cursor: not-allowed;
        }

        .sudoku-page {
          max-width: 520px;
          margin: 30px auto;
          padding: 20px;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          user-select: none;
          text-align: center;
          background: linear-gradient(45deg, #74ebd5, #9face6);
          border-radius: 15px;
          box-shadow: 0 0 25px rgba(0, 0, 0, 0.3);
        }

        .number-pad {
          display: flex;
          justify-content: center;
          margin-bottom: 20px;
          gap: 10px;
          flex-wrap: wrap;
        }

        .number-button {
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: white;
          border: none;
          border-radius: 6px;
          padding: 10px 16px;
          cursor: pointer;
          font-size: 18px;
          box-shadow: 0 4px 8px rgba(118, 75, 162, 0.3);
          transition: background 0.3s ease;
          user-select: none;
        }

        .number-button:disabled {
          background: #bbb;
          cursor: not-allowed;
          box-shadow: none;
          color: #555;
        }

        .number-button:hover:enabled {
          background: linear-gradient(135deg, #764ba2, #667eea);
        }

        .number-button.clear {
          background: linear-gradient(135deg, #ff6b6b, #f06595);
          box-shadow: 0 4px 8px rgba(240, 101, 149, 0.5);
        }

        .number-button.clear:hover:enabled {
          background: linear-gradient(135deg, #f06595, #ff6b6b);
        }

        .number-button.reveal {
          background: linear-gradient(135deg, #6bd5ff, #4287f5);
          box-shadow: 0 4px 8px rgba(66, 135, 245, 0.5);
        }

        .number-button.reveal:hover:enabled {
          background: linear-gradient(135deg, #4287f5, #6bd5ff);
        }

        h1 {
          margin-bottom: 10px;
          color: #222;
          text-shadow: 0 0 1px #aaa;
        }

        .sudoku-grid {
          display: grid;
          grid-template-rows: repeat(9, 1fr);
          gap: 0;
          border: 3px solid #333;
          border-radius: 10px;
          background: #fdfdfd;
          box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
        }

        .sudoku-row {
          display: grid;
          grid-template-columns: repeat(9, 1fr);
        }

        .sudoku-cell {
          border: 1px solid #bbb;
          width: 52px;
          height: 52px;
          line-height: 52px;
          font-size: 26px;
          text-align: center;
          vertical-align: middle;
          cursor: pointer;
          position: relative;
          transition: background 0.3s, box-shadow 0.3s;
          box-sizing: border-box;
          box-shadow: inset 1px 1px 3px rgba(255, 255, 255, 0.5);
        }

        .sudoku-cell:nth-child(3n) {
          border-right: 2px solid #444;
        }

        .sudoku-row:nth-child(3n) .sudoku-cell {
          border-bottom: 2px solid #444;
        }

        .starting-cell {
          font-weight: 700;
          color: #4a90e2;
          cursor: default;
          text-shadow: 0 0 3px #bbe1ff;
        }

        .starting-number {
          font-weight: 700;
          color: #4a90e2;
          user-select: none;
          display: inline-block;
          width: 100%;
        }

        .user-number {
          font-weight: 700;
          color: #0b3e02;
          user-select: none;
          display: inline-block;
          width: 100%;
        }

        .revealed-number {
          font-weight: 700;
          color: #d9534f;
          user-select: none;
          display: inline-block;
          width: 100%;
        }

        .selected-cell {
          box-shadow: 0 0 15px 3px #77c;
          background-color: #ddecff !important;
          cursor: pointer;
          user-select: none;
        }

        .error-cell {
          background-color: #f8d7da !important;
          color: #721c24 !important;
          font-weight: 900;
          cursor: pointer;
          box-shadow: 0 0 15px 3px #ff6f61 !important;
        }
      `}</style>
    </div>
  )
}
