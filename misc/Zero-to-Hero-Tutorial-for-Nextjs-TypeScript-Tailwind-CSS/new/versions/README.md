# Sudoku Next.js Versions

This directory contains multiple versions of a Sudoku game implemented in Next.js. Each file demonstrates incremental improvements and feature additions made during a development session.

## Versions

- **sudoku_v1_basic.js**
  - Initial attractive Sudoku with input fields for numbers
  - 9x9 grid, basic validation on inputs

- **sudoku_v2_random.js**
  - Add random puzzle selection from predefined puzzles

- **sudoku_v3_generator_colored.js**
  - Add puzzle generator using backtracking solver
  - Colorful 3x3 block backgrounds
  - Countdown timer and difficulty selector integrated
  - Styled for colorful user experience

- **sudoku_v4_click_keypad.js**
  - Replace input fields with clickable number keypad
  - Number keypad appears above grid

- **sudoku_v5_fixed_keypad_reveal.js**
  - Number keypad permanently visible above grid
  - Add "Reveal" button to show correct number in selected cell

- **sudoku_v6_title_and_contrast.js**
  - Move title above keypad
  - Improve contrast of user entered numbers (darker green)

- **sudoku_v7_lighter_prefilled_blue.js**
  - Soften blue color for prefilled starting numbers

- **sudoku_v8_revealed_numbers_red.js**
  - Revealed numbers displayed in distinct light red color

- **sudoku_v9_difficulty_selector.js**
  - Add difficulty selector dropdown (Easy, Medium, Hard)
  - Difficulty affects generated puzzle complexity

- **sudoku_v10_timer_and_title.js**
  - Add countdown timer bar starting from 5, 10, or 15 minutes (default 5)
  - Timer disables input on expiry
  - Title updated to "Sree's Sudoku Advanced"

- **sudoku_v11_timer_per_difficulty.js**
  - Timer duration dynamically updates per difficulty:
    - Easy: 5 minutes
    - Medium: 10 minutes
    - Hard: 15 minutes

---

Each version file is a self-contained React component for the Sudoku game.

You may integrate them into a Next.js project under the `pages` directory to test or build upon.

---

For additional help or packaging, feel free to ask.
