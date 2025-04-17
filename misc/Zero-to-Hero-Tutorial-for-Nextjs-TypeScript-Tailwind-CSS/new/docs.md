# Next.js Sudoku Application - File Structure and Overview

This document explains the main folders and files of the Next.js Sudoku application for beginners.

---

## Root Directory

- **package.json**: Contains metadata about the project, including dependencies (Next.js, React) and scripts to run, build, and start the app.
- **Dockerfile**: Defines how to containerize the application using Node.js. It installs dependencies, builds the app, and starts the server.
- **docker-compose.yml**: Configuration file to run the app's Docker container easily with predefined ports and restart policies.
- **next.config.js** (if present): Configuration file to customize Next.js build and runtime settings.

---

## styles Folder

- Contains global CSS files affecting the entire app.
- **globals.css**: Applies global styles such as fonts, background colors, link styles, and basic layout styles.

---

## pages Folder

This folder contains the main Next.js pages. Each file corresponds to a route.

- **_app.js**: The custom App component to initialize pages. It imports global CSS here so that styles apply across all pages.

- **index.js**: The root landing page displayed at `/`. Currently provides a welcome message and navigation link to the Sudoku game.

- **sudoku.js**: The main Sudoku game page.
  - Implements the interactive Sudoku board using React.
  - Contains the game logic: puzzle generation, validation, user input handling, timer, difficulty selector.
  - Contains all styles scoped to the Sudoku page using styled-jsx.

---

## How It Works

- When you run the app (via `npm run dev` or Docker container), Next.js serves pages based on `pages` directory.
- Global styles from `styles/globals.css` are loaded on every page.
- The Sudoku page provides a colorful interactive Sudoku game:
  - Generates puzzles of differing difficulty.
  - Allows number entry via a clickable keypad.
  - Shows errors and allows revealing correct answers.
  - Has a countdown timer.

---

This simple, self-contained structure makes it easy to understand and modify the application.

Please ask if you want examples or more detailed guidance on any part.
