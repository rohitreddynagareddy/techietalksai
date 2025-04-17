import React, { useState, useEffect } from 'react'

const puzzles = [
  [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
  ],
  [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0],
  ],
  [
    [0, 2, 0, 6, 0, 8, 0, 0, 0],
    [5, 8, 0, 0, 0, 9, 7, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [3, 7, 0, 0, 0, 0, 5, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 8, 0, 0, 0, 0, 1, 3],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 9, 8, 0, 0, 0, 3, 6],
    [0, 0, 0, 3, 0, 6, 0, 9, 0],
  ]
]

function getRandomPuzzle() {
  const index = Math.floor(Math.random() * puzzles.length)
  return puzzles[index]
}

function isNumberValid(grid, row, col, num) {
  for (let c = 0; c < 9; c++) {
    if (grid[row][c] === num) return false;
  }
  for (let r = 0; r < 9; r++) {
    if (grid[r][col] === num) return false;
  }
  const boxRow = Math.floor(row / 3) * 3;
  const boxCol = Math.floor(col / 3) * 3;
  for (let r = boxRow; r < boxRow + 3; r++) {
    for (let c = boxCol; c < boxCol + 3; c++) {
      if (grid[r][c] === num) return false;
    }
  }
  return true;
}

export default function Sudoku() {
  const [startingPuzzle, setStartingPuzzle] = useState(() => getRandomPuzzle());
  const [grid, setGrid] = useState(() => startingPuzzle.map(row => row.slice()));
  const [selected, setSelected] = useState(null); // [row, col]
  const [errorCells, setErrorCells] = useState([]); // array of [row,col]

  function handleCellClick(row, col) {
    setSelected([row, col]);
    setErrorCells([]);
  }

  function handleChange(e) {
    if (!selected) return;
    const val = e.target.value;
    if (val === '') {
      updateGrid(0);
      return;
    }
    const num = parseInt(val);
    if (num >= 1 && num <= 9) {
      updateGrid(num);
    }
  }

  function updateGrid(num) {
    setGrid((oldGrid) => {
      const newGrid = oldGrid.map(row => row.slice());
      const [r, c] = selected;
      // Only allow edit if starting cell was empty
      if (startingPuzzle[r][c] !== 0) return oldGrid;
      newGrid[r][c] = num;
      return newGrid;
    });
  }

  useEffect(() => {
    let errs = [];
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const val = grid[r][c];
        if (val !== 0) {
          const temp = grid[r][c];
          grid[r][c] = 0;
          if (!isNumberValid(grid, r, c, temp)) {
            errs.push([r, c]);
          }
          grid[r][c] = temp;
        }
      }
    }
    setErrorCells(errs);
  }, [grid]);

  function isErrorCell(r, c) {
    return errorCells.some(([rr, cc]) => rr === r && cc === c);
  }

  return (
    <div className="sudoku-page">
      <h1>Sudoku Game</h1>
      <div className="sudoku-grid">
        {grid.map((row, r) => (
          <div key={r} className="sudoku-row">
            {row.map((cell, c) => {
              const isStarting = startingPuzzle[r][c] !== 0;
              const isSelected = selected && selected[0] === r && selected[1] === c;
              const cellError = isErrorCell(r,c);
              return (
                <div
                  key={c}
                  className={`sudoku-cell ${isStarting ? 'starting-cell' : ''} ${isSelected ? 'selected-cell' : ''} ${cellError ? 'error-cell' : ''}`}
                  onClick={() => handleCellClick(r, c)}
                >
                  {isStarting ? cell : (
                    isSelected ? (
                      <input
                        type="text"
                        maxLength={1}
                        value={cell === 0 ? '' : cell}
                        onChange={handleChange}
                        onClick={(e) => e.stopPropagation()}
                        autoFocus
                      />
                    ) : (
                      cell !== 0 ? cell : ''
                    )
                  )}
                </div>
              )
            })}
          </div>
        ))}
      </div>
      <style jsx>{`
        .sudoku-page {
          max-width: 480px;
          margin: 30px auto;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          user-select: none;
          text-align: center;
        }
        h1 {
          margin-bottom: 15px;
          color: #333;
        }
        .sudoku-grid {
          display: grid;
          grid-template-rows: repeat(9, 1fr);
          gap: 0;
          border: 3px solid #333;
          border-radius: 10px;
          background: #fdfdfd;
          box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }
        .sudoku-row {
          display: grid;
          grid-template-columns: repeat(9, 1fr);
        }
        .sudoku-cell {
          border: 1px solid #bbb;
          width: 48px;
          height: 48px;
          line-height: 48px;
          font-size: 24px;
          text-align: center;
          vertical-align: middle;
          cursor: pointer;
          position: relative;
          background: #fff;
          transition: background 0.3s, box-shadow 0.3s;
          box-sizing: border-box;
        }
        .sudoku-cell:nth-child(3n) {
          border-right: 2px solid #333;
        }
        .sudoku-row:nth-child(3n) .sudoku-cell {
          border-bottom: 2px solid #333;
        }
        .starting-cell {
          background: #e9ecef;
          font-weight: bold;
          color: #222;
          cursor: default;
        }
        .selected-cell {
          background: #cce5ff;
          box-shadow: inset 0 0 10px #66b0ff;
          cursor: text;
        }
        .error-cell {
          background: #f8d7da;
          color: #a94442;
          font-weight: bolder;
          cursor: text;
        }
        input {
          width: 100%;
          height: 100%;
          border: none;
          text-align: center;
          font-size: 24px;
          font-family: inherit;
          background: transparent;
          color: #333;
          outline: none;
          cursor: text;
          user-select: text;
        }
        input::-webkit-inner-spin-button,
        input::-webkit-outer-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
      `}</style>
    </div>
  )
}
