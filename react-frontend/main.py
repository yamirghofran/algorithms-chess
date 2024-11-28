from flask import Flask, request, jsonify
from typing import Dict, Tuple, List, Set
from flask_cors import CORS  # Import CORS
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

class ChessBoard:
    def __init__(self):
        self.current_player = 'white'
        self.board: Dict[Tuple[int, int], str] = {}
        self.initialize_board()
        self.move_history: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
        self.legal_moves: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self.captured_pieces: Dict[str, List[str]] = {"white": [], "black": []}
        self.last_move = None
        # Add tracking for pieces that have moved
        self.moved_pieces: Set[Tuple[int, int]] = set()
        self.generate_legal_moves()

    def initialize_board(self):
        # Initialize pawns
        for col in range(8):
            self.board[(1, col)] = 'white_pawn'
            self.board[(6, col)] = 'black_pawn'

        # Initialize other pieces
        pieces = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']
        for col, piece in enumerate(pieces):
            self.board[(0, col)] = f'white_{piece}'
            self.board[(7, col)] = f'black_{piece}'

    def make_move(self, start: Tuple[int, int], end: Tuple[int, int], promotion: str = None) -> bool:
        print(f"Making move from {start} to {end}")  # Debug log
        if (start, end) not in self.legal_moves:
            print(f"Move {start} to {end} not in legal moves: {self.legal_moves}")  # Debug log
            return False

        moving_piece = self.board[start]
        print(f"Moving piece: {moving_piece}")  # Debug log
        captured_piece = None

        # Handle castling
        if moving_piece.endswith('king') and abs(end[1] - start[1]) == 2:
            # Kingside castling
            if end[1] > start[1]:
                rook_start = (start[0], 7)
                rook_end = (start[0], 5)
            # Queenside castling
            else:
                rook_start = (start[0], 0)
                rook_end = (start[0], 3)
            # Move the rook
            self.board[rook_end] = self.board.pop(rook_start)
            self.moved_pieces.add(rook_start)

        # Handle en passant capture
        if moving_piece.endswith('pawn'):
            if end[1] != start[1] and end not in self.board:  # Diagonal move to empty square
                if self.last_move:
                    last_start, last_end, last_piece = self.last_move
                    if (last_piece.endswith('pawn') and 
                        abs(last_start[0] - last_end[0]) == 2 and  # Last move was a two-square pawn advance
                        last_end[1] == end[1] and  # Same file as target square
                        last_end[0] == start[0]):  # Adjacent rank
                        # Remove the captured pawn
                        captured_piece = self.board.pop(last_end)
                        self.captured_pieces[self.current_player].append(captured_piece)
        
        # Handle regular captures
        if end in self.board:
            captured_piece = self.board[end]
            self.captured_pieces[self.current_player].append(captured_piece)

        # Make the move
        self.board.pop(start)
        self.board[end] = moving_piece
        self.moved_pieces.add(start)  # Track that this piece has moved
        self.move_history.append((start, end, captured_piece))
        self.last_move = (start, end, moving_piece)  # Update last_move

        # Handle pawn promotion
        if moving_piece.endswith('pawn'):
            if (self.current_player == 'white' and end[0] == 7) or (self.current_player == 'black' and end[0] == 0):
                if promotion in ['queen', 'rook', 'bishop', 'knight']:
                    self.board[end] = f'{self.current_player}_{promotion}'
                else:
                    self.board[end] = f'{self.current_player}_queen'

        # Update current player and regenerate legal moves for the new player
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        print(f"Current player after move: {self.current_player}")  # Debug log
        
        # Clear and regenerate legal moves for the new current player
        self.legal_moves.clear()
        self.generate_legal_moves()
        print(f"Legal moves after move: {self.legal_moves}")  # Debug log
        
        return True

    def undo_move(self) -> bool:
        if not self.move_history:
            return False

        start, end, captured_piece = self.move_history.pop()
        moving_piece = self.board[end]
        
        # Handle castling undo
        if moving_piece.endswith('king') and abs(end[1] - start[1]) == 2:
            # Kingside castling
            if end[1] > start[1]:
                rook_start = (start[0], 7)
                rook_end = (start[0], 5)
            # Queenside castling
            else:
                rook_start = (start[0], 0)
                rook_end = (start[0], 3)
            # Move the rook back
            self.board[rook_start] = self.board.pop(rook_end)
            self.moved_pieces.remove(rook_start)

        # Regular move undo
        self.board[start] = self.board.pop(end)
        if captured_piece:
            capturing_color = 'white' if self.current_player == 'black' else 'black'
            self.captured_pieces[capturing_color].pop()
            self.board[end] = captured_piece

        # Remove the moved piece tracking
        self.moved_pieces.remove(start)
        
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        self.generate_legal_moves()
        return True

    def generate_legal_moves(self):
        self.legal_moves.clear()
        is_in_check = self.is_in_check(self.current_player)
        
        # Create a list of positions and pieces to avoid modifying during iteration
        positions_and_pieces = [(pos, piece) for pos, piece in self.board.items() 
                              if piece.startswith(self.current_player)]
        
        for pos, piece in positions_and_pieces:
            potential_moves = self.get_piece_moves(pos, piece)
            for move in potential_moves:
                if self.try_move(move[0], move[1]):
                    self.legal_moves.add(move)
        
        # If in check, filter out moves that don't resolve the check
        if is_in_check:
            self.legal_moves = {move for move in self.legal_moves 
                              if not self.is_in_check_after_move(move)}
        
        #print(f"Generated legal moves: {self.legal_moves}")

    def get_piece_moves(self, pos: Tuple[int, int], piece: str, ignore_turn: bool = False, for_check: bool = False) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        moves = set()
        row, col = pos

        if piece.endswith('pawn'):
            direction = 1 if piece.startswith('white') else -1
            
            # Normal forward movement
            if 0 <= row + direction < 8 and (row + direction, col) not in self.board:
                moves.add((pos, (row + direction, col)))
                
                # Initial two-square move
                if ((row == 1 and piece.startswith('white')) or 
                    (row == 6 and piece.startswith('black'))):
                    if (row + 2 * direction, col) not in self.board:
                        moves.add((pos, (row + 2 * direction, col)))
            
            # Regular diagonal captures
            for capture_col in [col - 1, col + 1]:
                if 0 <= capture_col < 8 and 0 <= row + direction < 8:
                    target = (row + direction, capture_col)
                    if target in self.board and not self.board[target].startswith(piece.split('_')[0]):
                        moves.add((pos, target))
            
            # En passant captures
            if self.last_move:
                last_start, last_end, last_piece = self.last_move
                if (last_piece.endswith('pawn') and 
                    abs(last_start[0] - last_end[0]) == 2 and  # Last move was a two-square pawn advance
                    last_end[1] in [col - 1, col + 1] and  # Adjacent file
                    last_end[0] == row):  # Same rank
                    moves.add((pos, (row + direction, last_end[1])))

        elif piece.endswith('rook'):
            # Implement rook movement logic
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Right, Left, Up, Down
                for step in range(1, 8):
                    new_row, new_col = row + step * direction[0], col + step * direction[1]
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target_piece = self.board.get((new_row, new_col))
                        if target_piece is None:
                            moves.add((pos, (new_row, new_col)))
                        elif not target_piece.startswith(piece.split('_')[0]):
                            moves.add((pos, (new_row, new_col)))
                            break
                        else:
                            break
                    else:
                        break

        elif piece.endswith('queen'):
            # Implement queen movement logic
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                for step in range(1, 8):
                    new_row, new_col = row + step * direction[0], col + step * direction[1]
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target_piece = self.board.get((new_row, new_col))
                        if target_piece is None:
                            moves.add((pos, (new_row, new_col)))
                        elif not target_piece.startswith(piece.split('_')[0]):
                            moves.add((pos, (new_row, new_col)))
                            break
                        else:
                            break
                    else:
                        break

        elif piece.endswith('king'):
            # Normal king moves
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                new_row, new_col = row + direction[0], col + direction[1]
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_piece = self.board.get((new_row, new_col))
                    if target_piece is None or not target_piece.startswith(piece.split('_')[0]):
                        moves.add((pos, (new_row, new_col)))

            # Add castling moves only if not checking for attacks
            if not for_check and pos not in self.moved_pieces:
                # Kingside castling
                if (row, 7) not in self.moved_pieces:  # Rook hasn't moved
                    if all(self.board.get((row, col + i)) is None for i in [1, 2]):  # Path is clear
                        if not any(self.is_square_attacked((row, col + i), piece.split('_')[0]) for i in [0, 1, 2]):
                            moves.add((pos, (row, col + 2)))

                # Queenside castling
                if (row, 0) not in self.moved_pieces:  # Rook hasn't moved
                    if all(self.board.get((row, col - i)) is None for i in [1, 2, 3]):  # Path is clear
                        if not any(self.is_square_attacked((row, col - i), piece.split('_')[0]) for i in [0, 1, 2]):
                            moves.add((pos, (row, col - 2)))

        elif piece.endswith('bishop'):
            # Implement bishop movement logic
            for direction in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
                for step in range(1, 8):
                    new_row, new_col = row + step * direction[0], col + step * direction[1]
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target_piece = self.board.get((new_row, new_col))
                        if target_piece is None:
                            moves.add((pos, (new_row, new_col)))
                        elif not target_piece.startswith(piece.split('_')[0]):
                            moves.add((pos, (new_row, new_col)))
                            break
                        else:
                            break
                    else:
                        break

        elif piece.endswith('knight'):
            # Implement knight movement logic
            for direction in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                new_row, new_col = row + direction[0], col + direction[1]
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_piece = self.board.get((new_row, new_col))
                    if target_piece is None:
                        moves.add((pos, (new_row, new_col)))
                    elif not target_piece.startswith(piece.split('_')[0]):
                        moves.add((pos, (new_row, new_col)))

        return moves

    def print_board(self):
        for row in range(7, -1, -1):
            for col in range(8):
                piece = self.board.get((row, col), '.')
                #print(f'{piece:12}', end='')
            #print()

    def insufficient_material(self) -> bool:
        # Count the number of pieces for each player
        white_pieces = [piece for piece in self.board.values() if piece.startswith('white')]
        black_pieces = [piece for piece in self.board.values() if piece.startswith('black')]

        # Check for king vs king
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True

        # Check for king and bishop vs king or king and knight vs king
        if (len(white_pieces) == 2 and len(black_pieces) == 1) or (len(white_pieces) == 1 and len(black_pieces) == 2):
            for pieces in [white_pieces, black_pieces]:
                if len(pieces) == 2:
                    non_king_piece = [p for p in pieces if not p.endswith('king')][0]
                    if non_king_piece.endswith('bishop') or non_king_piece.endswith('knight'):
                        return True

        # Check for king and bishop vs king and bishop with bishops on same color
        if len(white_pieces) == 2 and len(black_pieces) == 2:
            white_bishop = next((p for p in white_pieces if p.endswith('bishop')), None)
            black_bishop = next((p for p in black_pieces if p.endswith('bishop')), None)
            if white_bishop and black_bishop:
                white_bishop_pos = next(pos for pos, piece in self.board.items() if piece == white_bishop)
                black_bishop_pos = next(pos for pos, piece in self.board.items() if piece == black_bishop)
                if (sum(white_bishop_pos) % 2) == (sum(black_bishop_pos) % 2):
                    return True
        return False

    def is_in_check(self, player: str) -> bool:
        king_pos = self.find_king(player)
        opponent = 'black' if player == 'white' else 'white'
        #print(f"Checking if {player} king at {king_pos} is in check")
        #print("Current board state:")
        #self.print_board()
        
        #print(f"Checking king at {king_pos}")
        opponent_moves = self.generate_player_moves(opponent, ignore_turn=True)
        for start, end in opponent_moves:
            if end == king_pos:
                piece = self.board[start]
                #print(f"Check detected by {piece} at {start}")
                return True
        
        #print("No check detected")
        return False

    def is_checkmate(self) -> bool:
        if not self.is_in_check(self.current_player):
            return False

        for start, end in self.legal_moves:
            if self.try_move(start, end):
                return False

        return True

    def is_stalemate(self) -> bool:
        if self.is_in_check(self.current_player):
            return False
        
        # Check if there are any legal moves available
        return len(self.legal_moves) == 0

    def is_draw(self) -> bool:
        # Check if the game is a draw by repetition
        #if self.move_history.count(self.move_history[0]) >= 3:
        #    return True
        if self.insufficient_material():
            return True
        return False

    def generate_player_moves(self, player: str, ignore_turn: bool = False) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        moves = set()
        for pos, piece in self.board.items():
            if piece.startswith(player):
                piece_moves = self.get_piece_moves(pos, piece, ignore_turn)
                #print(f"Moves for {piece} at {pos}: {piece_moves}")
                moves.update(piece_moves)
        return moves

    def try_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        # Check if the start position exists before trying to pop
        if start not in self.board:
            return False

        original_board = self.board.copy()
        piece = self.board.pop(start)
        captured_piece = self.board.get(end)
        self.board[end] = piece

        is_safe = not self.is_in_check(self.current_player)

        self.board = original_board
        if captured_piece:
            self.board[end] = captured_piece
        return is_safe

    def find_king(self, player: str) -> Tuple[int, int]:
        for pos, piece in self.board.items():
            if piece == f'{player}_king':
                return pos
        raise ValueError(f"King not found for player {player}")

    def reset(self):
        self.board.clear()
        self.initialize_board()
        self.move_history.clear()
        self.legal_moves.clear()
        self.generate_legal_moves()
        self.current_player = 'white'
        self.captured_pieces = {"white": [], "black": []}
        self.moved_pieces.clear()

    def is_in_check_after_move(self, move):
        start, end = move
        original_board = self.board.copy()
        piece = self.board.pop(start)
        captured_piece = self.board.get(end)
        self.board[end] = piece

        is_still_in_check = self.is_in_check(self.current_player)

        self.board = original_board
        return is_still_in_check

    def is_square_attacked(self, pos: Tuple[int, int], player: str) -> bool:
        opponent = 'black' if player == 'white' else 'white'
        for piece_pos, piece in self.board.items():
            if piece.startswith(opponent):
                moves = self.get_piece_moves(piece_pos, piece, ignore_turn=True, for_check=True)
                if pos in [move[1] for move in moves]:
                    return True
        return False

    def evaluate_position(self) -> float:
        """Simple position evaluation based on material"""
        piece_values = {
            'pawn': 1,
            'knight': 3,
            'bishop': 3,
            'rook': 5,
            'queen': 9,
            'king': 0
        }
        
        score = 0
        for piece in self.board.values():
            color, piece_type = piece.split('_')
            value = piece_values[piece_type]
            if color == 'white':
                score += value
            else:
                score -= value
        return score

    def make_cpu_move(self, depth=3) -> bool:
        """Makes a move for CPU by exploring possible moves"""
        print("CPU is making a move")
        print(f"Current player before CPU move: {self.current_player}")
        
        # Ensure legal moves are up to date for the current player
        self.generate_legal_moves()
        moves = list(self.legal_moves)
        print(f"Available legal moves: {moves}")  # Debug log
        
        if not moves:
            print("No legal moves available")
            return False

        best_move = None
        best_score = float('-inf') if self.current_player == 'white' else float('inf')
        
        for move in moves:
            # Create a copy of the board state
            original_board = self.board.copy()
            original_moves = self.legal_moves.copy()  # Save legal moves
            original_player = self.current_player     # Save current player
            
            # Try the move
            start, end = move
            # Convert tuples to ensure they're the same format
            start = tuple(map(int, start))  # Convert to int tuple
            end = tuple(map(int, end))      # Convert to int tuple
            
            # Make the move on the temporary board
            captured_piece = self.board.get(end)
            self.board[end] = self.board.pop(start)
            self.current_player = 'black' if self.current_player == 'white' else 'white'
            
            # Explore this move's possible outcomes
            score = self.explore_moves(depth - 1, [])
            
            # Restore the board state
            self.board = original_board
            self.legal_moves = original_moves
            self.current_player = original_player
            
            # Update best move based on score
            if self.current_player == 'white':
                if score > best_score:
                    best_score = score
                    best_move = (start, end)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (start, end)

        if best_move:
            print(f"CPU choosing move: {best_move}")
            start, end = best_move
            success = self.make_move(start, end)
            print(f"Move success: {success}")
            return success
            
        return False

    def explore_moves(self, depth: int, move_sequence: list) -> float:
        """Explores possible moves using a tree structure and backtracking"""
        # Base case: reached desired depth or game is over
        if depth == 0 or self.is_checkmate() or self.is_stalemate():
            return self.evaluate_position()

        # Generate legal moves for current position
        self.generate_legal_moves()
        moves = list(self.legal_moves)
        if not moves:
            return self.evaluate_position()

        scores = []
        original_player = self.current_player
        
        for move in moves:
            # Make move
            start, end = move
            captured_piece = self.board.get(end)
            moving_piece = self.board.get(start)
            
            if moving_piece is None:  # Skip if the piece doesn't exist
                continue
                
            self.board[end] = moving_piece
            self.board.pop(start)
            move_sequence.append(move)
            self.current_player = 'black' if self.current_player == 'white' else 'white'

            # Recursively explore this path
            score = self.explore_moves(depth - 1, move_sequence)
            scores.append(score)

            # Backtrack: undo move
            self.board[start] = moving_piece
            self.board.pop(end)
            if captured_piece:
                self.board[end] = captured_piece
            move_sequence.pop()
            self.current_player = original_player

        # Return best score based on current player
        if not scores:
            return 0
        return max(scores) if original_player == 'white' else min(scores)

# Create a global instance of ChessBoard
chess_board = ChessBoard()

def get_game_state():
    """Helper function to get current game state"""
    board_serializable = {f"{pos[0]},{pos[1]}": piece for pos, piece in chess_board.board.items()}
    
    return {
        'board': board_serializable,
        'currentPlayer': chess_board.current_player,
        'capturedPieces': chess_board.captured_pieces,
        'inCheck': chess_board.is_in_check(chess_board.current_player),
        'inCheckmate': chess_board.is_checkmate(),
        'inStalemate': chess_board.is_stalemate(),
        'isDraw': chess_board.is_draw()
    }

@app.route('/move/player', methods=['POST'])
def make_player_move():
    data = request.json
    start = tuple(data['start'])
    end = tuple(data['end'])
    
    # Make player's move
    player_success = chess_board.make_move(start, end, data.get('promotion'))
    
    return jsonify({
        'success': player_success,
        'message': 'Invalid move' if not player_success else None,
        'gameState': get_game_state()
    })

@app.route('/move/cpu', methods=['POST'])
def make_cpu_move():
    # Make CPU move
    cpu_success = chess_board.make_cpu_move()
    
    return jsonify({
        'success': cpu_success,
        'gameState': get_game_state()
    })

@app.route('/board', methods=['GET'])
def get_board():
    return jsonify(get_game_state())

@app.route('/legal-moves', methods=['GET'])
def get_legal_moves():
    start_time = time.time()
    legal_moves = list(chess_board.legal_moves)
    end_time = time.time()
    print(f"Legal moves endpoint took {end_time - start_time:.4f} seconds")
    return jsonify(legal_moves)

@app.route('/undo', methods=['POST'])
def undo_move():
    success = chess_board.undo_move()
    return jsonify({'success': success})

@app.route('/reset', methods=['POST'])
def reset_board():
    global chess_board
    chess_board.reset()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
