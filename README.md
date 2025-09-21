# Algorithms Project - Chess
https://github.com/user-attachments/assets/defcabd3-a500-4985-9364-01a622929080

Yousef Amirghofran, Lea Aboujaoudé, Diana Cordovez, Kareem Ramil Jamil


## Overview

This project implements a fully functional chess game featuring an AI opponent, complete with a React frontend and Flask backend. The application demonstrates various algorithms and data structures commonly used in game development and computer science.

## Key Features

- **Complete Chess Implementation**: All standard chess rules including castling, en passant, pawn promotion, check, checkmate, stalemate, and draw conditions
- **AI Opponent**: CPU player using the minimax algorithm with alpha-beta pruning
- **Real-time Gameplay**: Drag-and-drop interface with real-time game state updates
- **Captured Pieces Tracking**: Binary search tree implementation for efficient piece management
- **Leaderboard System**: Quick sort algorithm for ranking players
- **Modern Tech Stack**: React frontend with TypeScript, Flask backend with Python

## Algorithms and Data Structures

### Backend Algorithms

#### 1. Minimax Algorithm with Alpha-Beta Pruning

- **Location**: `ChessBoard.explore_moves()` and `ChessBoard.make_cpu_move()` in `main.py`
- **Purpose**: Powers the AI opponent by exploring possible move sequences to find optimal moves
- **Time Complexity**: O(b^d) where b is the branching factor and d is the search depth
- **Implementation Details**:
  - Recursive tree search exploring moves up to depth 3
  - Alpha-beta pruning optimization to reduce search space
  - Position evaluation based on material advantage and game state

#### 2. Binary Search Tree for Captured Pieces

- **Location**: `CapturedPiecesTree` class in `main.py`
- **Purpose**: Efficiently track and manage captured pieces during gameplay
- **Time Complexity**:
  - Insert: O(log n) average case, O(n) worst case
  - Get total value: O(n)
  - In-order traversal: O(n)
- **Implementation Details**:
  - Custom BST implementation for storing captured pieces by their material value
  - Efficient insertion and traversal operations
  - Used to display captured pieces in sorted order by value

#### 3. Quick Sort Algorithm

- **Location**: `quick_sort_leaderboard()` in `leaderboard.py`
- **Purpose**: Sort leaderboard entries by score for ranking display
- **Time Complexity**: O(n log n) average case, O(n²) worst case
- **Implementation Details**:
  - In-place sorting algorithm
  - Uses partition scheme to divide array around pivot
  - Optimized for leaderboard data with numerical scores

#### 4. Chess-Specific Algorithms

- **Move Generation**: O(n²) time complexity for generating all legal moves
- **Check Detection**: O(n²) time complexity for determining if a king is in check
- **Game State Evaluation**: O(n) time complexity for position assessment
- **Insufficient Material Detection**: O(n) time complexity for draw conditions

### Frontend Technologies

- **React Query**: For efficient API state management and caching
- **TypeScript**: For type safety and better development experience
- **Tailwind CSS**: For responsive and modern UI design
- **React Drag and Drop**: For intuitive piece movement

## API Integration

### Flask Backend Endpoints

1. **GET /board** - Retrieve current game state

   - Returns: Board position, current player, captured pieces, game status
   - Time Complexity: O(n) average case

2. **POST /move/player** - Make a player move

   - Payload: `{start: [row, col], end: [row, col], promotion?: string}`
   - Triggers CPU move automatically after player move
   - Time Complexity: O(n) average case

3. **POST /move/cpu** - Trigger CPU move calculation

   - Uses minimax algorithm to determine best move
   - Time Complexity: O(b^d) exponential

4. **POST /reset** - Reset the game board

   - Time Complexity: O(1)

5. **GET /leaderboard** - Get sorted leaderboard

   - Uses quick sort for O(n log n) sorting
   - Time Complexity: O(n log n)

6. **POST /leaderboard/update** - Update player statistics
   - Time Complexity: O(n)

### React Query Integration

The frontend uses React Query for efficient API state management:

```typescript
// Query for game state
const { data: gameState } = useQuery({
  queryKey: ["gameState"],
  queryFn: async () => {
    const response = await axios.get("/api/board");
    return response.data;
  },
});

// Mutation for player moves
const playerMoveMutation = useMutation({
  mutationFn: (movePayload: MovePayload) => {
    return axios.post("/api/move/player", movePayload);
  },
  onSuccess: (response) => {
    queryClient.setQueryData(["gameState"], response.data.gameState);
    cpuMoveMutation.mutate(); // Automatically trigger CPU move
  },
});
```

## Key Functions and Classes

### Backend Core Classes

#### ChessBoard Class

- **Purpose**: Main game logic and state management
- **Key Methods**:
  - `make_move()`: O(n²) - Validates and executes moves
  - `generate_legal_moves()`: O(n²) - Creates all possible legal moves
  - `is_in_check()`: O(n²) - Determines if king is under attack
  - `explore_moves()`: O(b^d) - Minimax algorithm implementation
  - `evaluate_position()`: O(n) - Position scoring for AI

#### CapturedPiecesTree Class

- **Purpose**: Binary search tree for captured pieces management
- **Key Methods**:
  - `insert()`: O(log n) - Add captured piece to tree
  - `get_total_value()`: O(n) - Calculate total material value
  - `get_pieces_by_value()`: O(n) - Retrieve sorted pieces

#### MoveNode Class

- **Purpose**: Tree node for minimax algorithm
- **Key Methods**:
  - `add_child()`: O(1) - Add child move
  - `get_path_to_root()`: O(d) - Get move sequence

### Frontend Components

#### ChessBoard Component

- **Purpose**: Main game interface with drag-and-drop functionality
- **Key Features**:
  - Real-time game state updates
  - Pawn promotion dialog
  - Captured pieces display
  - Move validation and feedback

#### Leaderboard Component

- **Purpose**: Display sorted player rankings
- **Key Features**:
  - Real-time leaderboard updates
  - Score calculation and ranking
  - Integration with game results

## Setup and Installation

### Python Backend

1. **Create virtual environment**:

   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:

   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Flask backend**:
   ```bash
   cd python-backend
   python main.py
   ```
   Backend will run on `http://localhost:5000`

### React Frontend

1. **Install Bun**:

   - Visit [bun.sh](https://bun.sh/) for installation instructions

2. **Install dependencies**:

   ```bash
   cd react-frontend
   bun install
   ```

3. **Start development server**:

   ```bash
   bun run dev
   ```

   Frontend will run on `http://localhost:5173`

4. **Open browser**: Navigate to `http://localhost:5173`

## Project Structure

```
algorithms/
├── python-backend/
│   ├── main.py              # Flask app and chess logic
│   ├── leaderboard.py       # Quick sort implementation
│   └── requirements.txt     # Python dependencies
└── react-frontend/
    ├── src/
    │   ├── components/
    │   │   ├── ui/          # UI components
    │   │   └── Leaderboard.tsx
    │   ├── routes/
    │   │   └── chess-board.tsx
    │   └── App.tsx          # Main app component
    └── public/pieces/       # Chess piece SVGs
```

## Dependencies

### Backend

- Flask 3.0.2: Web framework
- Flask-Cors 4.0.0: Cross-origin resource sharing
- Werkzeug 3.0.1: WSGI utility library

### Frontend

- React 18.3.1: UI library
- TypeScript 5.5.3: Type safety
- React Query 5.59.13: API state management
- Tailwind CSS 3.4.13: Styling framework
- Axios 1.7.7: HTTP client

## Performance Considerations

- **AI Move Calculation**: Limited to depth 3 to maintain responsive gameplay
- **Alpha-Beta Pruning**: Reduces minimax search space significantly
- **Efficient Data Structures**: BST for captured pieces, sets for move tracking
- **React Query Caching**: Reduces unnecessary API calls
- **Optimized Re-renders**: Proper state management to minimize component updates

## Known Issues

- Reset board button may require double-click in some cases
- CPU move calculation can be slow on older hardware due to exponential time complexity

## Future Improvements

- Increase AI search depth for stronger play
- Add opening book for varied CPU play
- Implement additional chess variants
- Add game history and move notation
- Mobile-responsive design improvements

