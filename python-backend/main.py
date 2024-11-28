from flask import Flask, request, jsonify
from typing import Dict, Tuple, List, Set
from flask_cors import CORS  # Import CORS
import time
import random
import csv
from leaderboard import LeaderboardEntry, quick_sort_leaderboard

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

class MoveNode:
    def __init__(self, board_state: dict, move: tuple = None, parent=None):
        # O(1) for initializing a new MoveNode
        # The function assigns the provided values to the node's attributes, which is done in constant time.
        self.board_state = board_state
        self.move = move  # (start, end) tuple
        self.parent = parent
        self.children = []
        self.score = None
        
    def add_child(self, child_node):
        # O(1) for adding a child node
        # The function appends the provided child node to the children list, which is done in constant time.
        self.children.append(child_node)
        
    def get_path_to_root(self):
        # O(d) where d is the depth of the node in the move tree
        # The function traverses from the current node to the root node by following the parent references.
        # In the worst case, the node is at the maximum depth of the move tree, resulting in a time complexity of O(d).
        path = []
        current = self
        while current.parent:
            path.append(current.move)
            current = current.parent
        return path[::-1]


class CapturedPieceNode:
    def __init__(self, piece: str, value: int):
        self.piece = piece
        self.value = value
        self.count = 1
        self.left = None
        self.right = None

class CapturedPiecesTree:
    def __init__(self):
        # O(1) for initializing the CapturedPiecesTree
        # The function initializes the root node and the piece_values dictionary, which is done in constant time.
        self.root = None
        self.piece_values = {
            'pawn': 1,
            'knight': 3,
            'bishop': 3,
            'rook': 5,
            'queen': 9
        }
        
    def insert(self, piece: str):
        # O(log n) on average and O(n) in the worst case for inserting a captured piece
        # The function determines the piece type and its corresponding value, then calls the recursive _insert function.
        # The _insert function traverses the binary search tree to find the appropriate position to insert the piece.
        # In a balanced tree, the time complexity is O(log n), where n is the number of captured pieces.
        # In the worst case of a skewed tree, the time complexity can be O(n).
        piece_type = piece.split('_')[1]
        value = self.piece_values[piece_type]
        self.root = self._insert(self.root, piece, value)
        
    def _insert(self, node, piece: str, value: int):
        # O(log n) on average and O(n) in the worst case for the recursive insertion helper function
        # The function recursively traverses the binary search tree to find the appropriate position to insert the piece.
        # In a balanced tree, the time complexity is O(log n), where n is the number of captured pieces.
        # In the worst case of a skewed tree, the time complexity can be O(n).
        if not node:
            return CapturedPieceNode(piece, value)
            
        if value < node.value:
            node.left = self._insert(node.left, piece, value)
        elif value > node.value:
            node.right = self._insert(node.right, piece, value)
        else:
            # Same value piece, increment count
            node.count += 1
        return node
        
    def get_total_value(self):
        # O(n) for getting the total value of captured pieces
        # The function calls the recursive _get_total_value function, which traverses the entire binary search tree.
        # Each node in the tree is visited once, resulting in a time complexity of O(n), where n is the number of captured pieces.
        return self._get_total_value(self.root)
        
    def _get_total_value(self, node):
        # O(n) for the recursive total value calculation helper function
        # The function recursively traverses the entire binary search tree, visiting each node once.
        # The time complexity is O(n), where n is the number of captured pieces.
        if not node:
            return 0
        return (node.value * node.count) + self._get_total_value(node.left) + self._get_total_value(node.right)
        
    def get_pieces_by_value(self):
        # O(n) for getting the captured pieces sorted by value
        # The function calls the recursive _inorder_traversal function, which performs an in-order traversal of the binary search tree.
        # Each node in the tree is visited once, resulting in a time complexity of O(n), where n is the number of captured pieces.
        pieces = []
        self._inorder_traversal(self.root, pieces)
        return pieces
        
    def _inorder_traversal(self, node, pieces):
        # O(n) for the recursive in-order traversal helper function
        # The function performs an in-order traversal of the binary search tree, visiting each node once.
        # The time complexity is O(n), where n is the number of captured pieces.
        if not node:
            return
        # Visit left subtree first (smaller values)
        self._inorder_traversal(node.left, pieces)
        # Add current node's pieces
        pieces.extend([node.piece] * node.count) # append node.piece node.count times
        # Visit right subtree last (bigger values)
        self._inorder_traversal(node.right, pieces)

class ChessBoard:
    def __init__(self):
        self.current_player = 'white'
        self.board: Dict[Tuple[int, int], str] = {}
        self.initialize_board()  # O(1)
        self.move_history: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
        self.legal_moves: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self.captured_pieces_trees = {
            "white": CapturedPiecesTree(),
            "black": CapturedPiecesTree()
        }
        self.last_move = None
        self.moved_pieces: Set[Tuple[int, int]] = set()
        self.generate_legal_moves()  # O(n^2) where n is the number of pieces

    def initialize_board(self):
        # O(1) since the board setup is constant
        # The function sets up the initial board state with a fixed number of operations, regardless of the board size.
        for col in range(8):
            self.board[(1, col)] = 'white_pawn'
            self.board[(6, col)] = 'black_pawn'

        # Initialize other pieces
        pieces = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']
        for col, piece in enumerate(pieces):
            self.board[(0, col)] = f'white_{piece}'
            self.board[(7, col)] = f'black_{piece}'

    def make_move(self, start: Tuple[int, int], end: Tuple[int, int], promotion: str = None) -> bool:
        # O(1) for checking and making a move, but O(n^2) for updating legal moves
        # where n is the number of pieces
        # Checking the validity of a move and updating the board state is done in constant time.
        # However, after making a move, the function calls generate_legal_moves(), which has a time complexity of O(n^2).
        if (start, end) not in self.legal_moves:
            #print(f"Move {start} to {end} not in legal moves: {self.legal_moves}")  # Debug log
            return False

        moving_piece = self.board[start]
        #print(f"Moving piece: {moving_piece}")  # Debug log
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
                        captured_piece = self.board.pop(last_end) # O(1)
                        self.captured_pieces_trees[self.current_player].insert(captured_piece) # O(log n)
        
        # Handle regular captures
        if end in self.board:
            captured_piece = self.board[end]
            self.captured_pieces_trees[self.current_player].insert(captured_piece)

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
        #print(f"Current player after move: {self.current_player}")  # Debug log
        
        # Clear and regenerate legal moves for the new current player
        self.legal_moves.clear()
        self.generate_legal_moves() # O(n^2) where n is the number of pieces
        #print(f"Legal moves after move: {self.legal_moves}")  # Debug log
        
        return True
    
    def generate_legal_moves(self):
        # O(n^2) where n is the number of pieces, due to nested loops
        # The function iterates over all the pieces on the board and calls get_piece_moves() for each piece.
        # get_piece_moves() itself has a worst-case time complexity of O(n) for certain piece types.
        # Therefore, the overall time complexity is O(n^2) due to the nested loops.
        self.legal_moves.clear()
        
        # Create a list of positions and pieces to avoid modifying during iteration
        positions_and_pieces = [(pos, piece) for pos, piece in self.board.items() 
                              if piece.startswith(self.current_player)]
        
        for pos, piece in positions_and_pieces:
            potential_moves = self.get_piece_moves(pos, piece)
            for move in potential_moves:
                if self.try_move(move[0], move[1]):
                    self.legal_moves.add(move)
        
        # filter out moves that walk into a check or don't resolve the check
        self.legal_moves = {move for move in self.legal_moves 
                            if not self.is_in_check_after_move(move)}
        
        #print(f"Generated legal moves: {self.legal_moves}")

    def get_piece_moves(self, pos: Tuple[int, int], piece: str, ignore_turn: bool = False, for_check: bool = False) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # O(1) for each piece type, but can be O(n) in the worst case for rooks, bishops, and queens (where n is the number of squares in the path)
        # The function determines the possible moves for a given piece based on its type.
        # For most piece types, the number of possible moves is constant.
        # However, for rooks, bishops, and queens, the function iterates along the respective paths until it reaches the end of the board or encounters another piece, resulting in a worst-case time complexity of O(n).
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

    def insufficient_material(self) -> bool:
        # O(n) where n is the number of pieces
        # The function iterates over all the pieces on the board to count the number of pieces for each player, resulting in a time complexity of O(n).
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
        # O(n^2) where n is the number of pieces, due to checking all opponent moves
        # The function finds the position of the king for the given player and then calls generate_player_moves() for the opponent.
        # generate_player_moves() iterates over all the opponent's pieces and calls get_piece_moves() for each piece, resulting in a time complexity of O(n^2).
        king_pos = self.find_king(player)
        opponent = 'black' if player == 'white' else 'white'
        #print(f"Checking if {player} king at {king_pos} is in check")
        #print("Current board state:")
        
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
        # O(n^2) where n is the number of pieces, due to checking all legal moves
        # The function first checks if the current player is in check using is_in_check(), which has a time complexity of O(n^2).
        # If the player is in check, the function iterates over all the legal moves and calls try_move() for each move to see if it resolves the check, resulting in a time complexity of O(n^2).
        if not self.is_in_check(self.current_player):
            return False

        for start, end in self.legal_moves:
            if self.try_move(start, end):
                return False

        return True

    def is_stalemate(self) -> bool:
        # O(n) where n is the number of legal moves
        # The function checks if the current player is in check using is_in_check(), which has a time complexity of O(n^2).
        # If the player is not in check, the function checks if there are any legal moves available, which is an O(n) operation.
        if self.is_in_check(self.current_player):
            return False
        
        # Check if there are any legal moves available
        return len(self.legal_moves) == 0

    def is_draw(self) -> bool:
        # O(n) where n is the number of pieces
        # The function calls insufficient_material(), which has a time complexity of O(n).
        if self.insufficient_material():
            return True
        return False

    def generate_player_moves(self, player: str, ignore_turn: bool = False) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # O(n^2) where n is the number of pieces
        # The function iterates over all the pieces on the board and calls get_piece_moves() for each piece belonging to the specified player.
        # get_piece_moves() itself has a worst-case time complexity of O(n) for certain piece types.
        # Therefore, the overall time complexity is O(n^2) due to the nested loops.
        moves = set()
        # Create a copy of the board items to avoid modification during iteration
        board_items = list(self.board.items())
        for pos, piece in board_items:
            if piece.startswith(player):
                piece_moves = self.get_piece_moves(pos, piece, ignore_turn)
                moves.update(piece_moves)
        return moves

    def try_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        # O(n^2) where n is the number of pieces, due to checking for check
        # The function makes a temporary move on a copy of the board and then calls is_in_check() to check if the move results in the player being in check.
        # is_in_check() has a time complexity of O(n^2), resulting in an overall time complexity of O(n^2) for try_move().
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
        # O(n) where n is the number of pieces
        # The function iterates over all the pieces on the board to find the position of the king for the specified player, resulting in a time complexity of O(n).
        for pos, piece in self.board.items():
            if piece == f'{player}_king':
                return pos
        raise ValueError(f"King not found for player {player}")

    def reset(self):
        # O(1) for resetting the board
        # The function clears the board and resets various game states, all of which are done in constant time.
        self.board.clear()
        self.initialize_board()
        self.move_history.clear()
        self.legal_moves.clear()
        self.generate_legal_moves()
        self.current_player = 'white'
        self.captured_pieces_trees = {
            "white": CapturedPiecesTree(),
            "black": CapturedPiecesTree()
        }
        self.moved_pieces.clear()

    def is_in_check_after_move(self, move):
        # O(n^2) where n is the number of pieces, due to checking for check
        # The function makes a temporary move on a copy of the board and then calls is_in_check() to check if the move results in the player being in check.
        # is_in_check() has a time complexity of O(n^2), resulting in an overall time complexity of O(n^2) for is_in_check_after_move().
        start, end = move
        
        # Check if start position exists
        if start not in self.board:
            return True  # Invalid move, treat as if still in check
        
        original_board = self.board.copy()
        piece = self.board.pop(start)
        captured_piece = self.board.get(end)
        self.board[end] = piece

        is_still_in_check = self.is_in_check(self.current_player)

        self.board = original_board
        return is_still_in_check

    def is_square_attacked(self, pos: Tuple[int, int], player: str) -> bool:
        # O(n^2) where n is the number of pieces, due to checking all opponent moves
        # The function iterates over all the pieces on the board and calls get_piece_moves() for each piece belonging to the opponent.
        # get_piece_moves() itself has a worst-case time complexity of O(n) for certain piece types.
        # Therefore, the overall time complexity is O(n^2) due to the nested loops.
        opponent = 'black' if player == 'white' else 'white'
        for piece_pos, piece in self.board.items():
            if piece.startswith(opponent):
                moves = self.get_piece_moves(piece_pos, piece, ignore_turn=True, for_check=True)
                if pos in [move[1] for move in moves]:
                    return True
        return False

    def evaluate_position(self) -> float:
        # O(n) where n is the number of pieces
        # The function iterates over all the pieces on the board to calculate the position evaluation based on material and piece position, resulting in a time complexity of O(n).
        """Position evaluation based on material, piece position and king attacks"""
        piece_values = {
            'pawn': 1,
            'knight': 3, 
            'bishop': 3,
            'rook': 5,
            'queen': 9,
            'king': 0
        }
        
        score = 0
        for pos, piece in self.board.items():
            row, col = pos
            color, piece_type = piece.split('_')
            value = piece_values[piece_type]
            
            # Base material value
            if color == 'white':
                score += value
                # Bonus for advancing pieces
                #score += row * 0.1
                # Extra bonus for attacking black king
                """if self.is_square_attacked(self.find_king('black'), 'white'):
                    score += 0.5"""
            else:
                score -= value
                # Bonus for advancing pieces
                #score -= (7-row) * 0.1
                # Extra bonus for attacking white king
                """if self.is_square_attacked(self.find_king('white'), 'black'):
                    score -= 0.5"""
                    
        return score

    def explore_moves(self, depth=3, highest_seen=float('-inf'), lowest_seen=float('inf'), maximizing=True) -> MoveNode:
        # The time complexity of explore_moves() depends on the depth of the search and the number of legal moves at each position.
        # In the worst case, the function explores all possible move sequences up to the specified depth (default is 3), resulting in an exponential time complexity.
        # The actual time complexity can be expressed as O(b^d), where b is the average branching factor (number of legal moves at each position) and d is the depth of the search.
        # The optimization helps reduce the number of positions evaluated, but the worst-case time complexity remains exponential.
        root = MoveNode(self.board.copy())
        
        if depth == 0:
            root.score = self.evaluate_position()
            return root
        
        moves = list(self.legal_moves)
        if not moves:
            root.score = self.evaluate_position()
            return root

        best_node = None
        best_score = float('-inf') if maximizing else float('inf')

        for move in moves:
            # Make move
            start, end = move
            captured_piece = self.board.get(end)
            piece = self.board.pop(start)
            self.board[end] = piece
            
            # Create child node and explore
            child = MoveNode(self.board.copy(), move, root)
            self.current_player = 'black' if self.current_player == 'white' else 'white'
            self.generate_legal_moves()
            
            # Recursively explore child positions
            explored = self.explore_moves(depth - 1, highest_seen, lowest_seen, not maximizing)
            child.score = explored.score
            
            # Undo move / backtrack
            self.board[start] = piece
            if captured_piece:
                self.board[end] = captured_piece
            else:
                self.board.pop(end)
            self.current_player = 'black' if self.current_player == 'white' else 'white'
            self.generate_legal_moves()
            
            # Update best score
            # As spoken with the professor, it was absolutely necessary to have this optimization here because without it, the game would be absolutely unplayable.
            # We have learned optimizations of this sort in class and have applied it to depth-first tree traversal.
            if maximizing:
                if child.score > best_score:
                    best_score = child.score
                    best_node = child
                highest_seen = max(highest_seen, best_score)
            else:
                if child.score < best_score:
                    best_score = child.score
                    best_node = child
                lowest_seen = min(lowest_seen, best_score)
                
            # skip exploring moves that can't be better
            if lowest_seen <= highest_seen:
                break
                
            root.add_child(child)
            
        root.score = best_score
        return best_node if best_node else root

    def make_cpu_move(self, depth=3) -> bool:
        # Time complexity:
        # - Average case: O(b^d) where b is the average branching factor and d is the search depth
        #   - O(b^d) for make_cpu_move() which calls explore_moves() with the specified depth
        #   - O(n) where n is the number of pieces for get_game_state() which serializes the board state
        #
        # - Worst case: O(b^d) 
        #   - Same as the average case
        #   - The worst case occurs when the explore_moves() algorithm explores all possible move sequences up to the maximum depth
        #   - O(n^2) for get_game_state() if it needs to check for checkmate/stalemate, but this is overshadowed by O(b^d)
        
        print("CPU is making a move")
        
        # Get best move with the explore_moves() algorithm - Black is minimizing player
        best_move = self.explore_moves(depth, maximizing=False)  # Changed to False for Black
        
        if not best_move or not best_move.move:
            return False
            
        start, end = best_move.move
        success = self.make_move(start, end)
        print(f"CPU chose move {start} to {end}, score: {best_move.score}, success: {success}")
        return success

    def get_material_advantage(self):
        # O(n) where n is the total number of captured pieces
        # The function retrieves the total value of captured pieces for both white and black players using the get_total_value() method of the CapturedPiecesTree.
        # The get_total_value() method performs an in-order traversal of the binary search tree, visiting each captured piece once, resulting in a time complexity of O(n).
        white_value = self.captured_pieces_trees["white"].get_total_value()
        black_value = self.captured_pieces_trees["black"].get_total_value()
        return white_value - black_value

# Create a global instance of ChessBoard
chess_board = ChessBoard()

def get_game_state():
    """Helper function to get current game state
    
    Time complexity:
    - Average case: O(n) where n is the number of pieces on the board
      - O(n) to serialize board dictionary
      - O(n) for get_pieces_by_value() which does inorder traversal of captured pieces trees
      - O(n) for is_in_check() which checks all pieces for attacks on king
      - O(n) for is_checkmate(), is_stalemate(), is_draw() which check legal moves
    
    - Worst case: O(n^2) if checking for checkmate/stalemate requires checking all possible moves
      for each piece
    """
    board_serializable = {f"{pos[0]},{pos[1]}": piece for pos, piece in chess_board.board.items()}
    
    return {
        'board': board_serializable,
        'currentPlayer': chess_board.current_player,
        'capturedPieces': {
            'white': chess_board.captured_pieces_trees['white'].get_pieces_by_value(),
            'black': chess_board.captured_pieces_trees['black'].get_pieces_by_value()
        },
        'inCheck': chess_board.is_in_check(chess_board.current_player),
        'inCheckmate': chess_board.is_checkmate(),
        'inStalemate': chess_board.is_stalemate(),
        'isDraw': chess_board.is_draw()
    }

@app.route('/move/player', methods=['POST'])
def make_player_move():
    """Handle player move request
    
    Time complexity:
    - Average case: O(n) where n is number of pieces
      - O(1) for data parsing and tuple conversion
      - O(n) for make_move() which updates board and generates legal moves
      - O(n) where n is number of captured pieces for get_pieces_by_value() which traverses captured pieces BST
      - O(n) for get_game_state() which serializes board state
    
    - Worst case: O(n^2) 
      - O(n^2) if make_move() needs to check all possible moves for each piece
      - O(n^2) if get_game_state() needs to check for checkmate/stalemate
    """
    data = request.json
    start = tuple(data['start'])
    end = tuple(data['end'])
    
    # Make player's move
    player_success = chess_board.make_move(start, end, data.get('promotion'))
    
    # Print sorted captured pieces from BST instead
    sorted_captured = {
        'white': chess_board.captured_pieces_trees['white'].get_pieces_by_value(),
        'black': chess_board.captured_pieces_trees['black'].get_pieces_by_value()
    }
    print(sorted_captured)
    
    return jsonify({
        'success': player_success,
        'message': 'Invalid move' if not player_success else None,
        'gameState': get_game_state()
    })

@app.route('/move/cpu', methods=['POST'])
def make_cpu_move():
    # Time complexity:
    # - Average case: O(b^d) where b is the average branching factor and d is the search depth
    #   - O(b^d) for make_cpu_move() which calls explore_moves() with the specified depth
    #   - explore_moves() has an exponential time complexity of O(b^d) for having to explore all possible moves
    #   - O(n) where n is the number of pieces for get_game_state() which serializes the board state
    #
    # - Worst case: O(b^d) 
    #   - Same as the average case, dominated by the exponential time complexity of explore_moves()
    #   - The worst case occurs when the explore_moves() algorithm explores all possible move sequences up to the maximum depth
    #   - O(n^2) for get_game_state() if it needs to check for checkmate/stalemate, but this is overshadowed by O(b^d)
    
    # Make CPU move
    cpu_success = chess_board.make_cpu_move()
    
    return jsonify({
        'success': cpu_success,
        'gameState': get_game_state()
    })

@app.route('/board', methods=['GET'])
def get_board():
    # Time complexity:
    # - Average case: O(n) where n is the number of pieces on the board
    #   - O(n) for get_game_state() which serializes the board state
    #
    # - Worst case: O(n^2)
    #   - O(n^2) for get_game_state() if it needs to check for checkmate/stalemate
    return jsonify(get_game_state())

@app.route('/legal-moves', methods=['GET'])
def get_legal_moves():
    # Time complexity:
    # - Average case: O(1)
    #   - O(1) for converting the set of legal moves to a list
    #
    # - Worst case: O(1)
    #   - The legal moves are already generated and stored in chess_board.legal_moves
    #   - Converting the set to a list has a constant time complexity
    start_time = time.time()
    legal_moves = list(chess_board.legal_moves)
    end_time = time.time()
    print(f"Legal moves endpoint took {end_time - start_time:.4f} seconds")
    return jsonify(legal_moves)

@app.route('/reset', methods=['POST'])
def reset_board():
    # Time complexity:
    # - Average case: O(1)
    #   - O(1) for resetting the chess_board instance
    #
    # - Worst case: O(1)
    #   - Resetting the chess_board instance has a constant time complexity
    global chess_board
    chess_board.reset()
    return jsonify({'success': True})

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    # Time complexity:
    # - Average case: O(n log n) where n is the number of leaderboard entries
    #   - O(n) for reading the leaderboard entries from the CSV file
    #   - O(n log n) for sorting the leaderboard entries using quick_sort_leaderboard()
    #
    # - Worst case: O(n^2)
    #   - O(n) for reading the leaderboard entries from the CSV file
    #   - O(n^2) if the quick_sort_leaderboard() algorithm degrades to O(n^2) in the worst case
    try:
        with open('leaderboard.csv', 'r') as f:
            entries = [LeaderboardEntry.from_csv_line(line) for line in f.readlines()[1:]]
        
        sorted_entries = quick_sort_leaderboard(entries)
        return jsonify([{
            'name': entry.name,
            'wins': entry.wins,
            'losses': entry.losses,
            'draws': entry.draws,
            'score': entry.score
        } for entry in sorted_entries])
    except FileNotFoundError:
        with open('leaderboard.csv', 'w') as f:
            f.write("name,wins,losses,draws,score\n")
        return jsonify([])

@app.route('/leaderboard/update', methods=['POST'])
def update_leaderboard():
    # Time complexity:
    # - Average case: O(n) where n is the number of leaderboard entries
    #   - O(n) for reading the leaderboard entries from the CSV file
    #   - O(n) for finding or creating the player entry
    #   - O(n) for writing the updated leaderboard entries to the CSV file
    #
    # - Worst case: O(n)
    #   - Same as the average case, as the operations have a linear time complexity
    data = request.json
    name = data['name']
    result = data['result']  # 'win', 'loss', or 'draw'
    
    entries = []
    try:
        with open('leaderboard.csv', 'r') as f:
            next(f)  # Skip header
            entries = [LeaderboardEntry.from_csv_line(line) for line in f.readlines()]
    except FileNotFoundError:
        pass
    
    # Find or create player entry
    player_entry = next((e for e in entries if e.name == name), LeaderboardEntry(name))
    if player_entry not in entries:
        entries.append(player_entry)
    
    # Update stats
    if result == 'win':
        player_entry.wins += 1
    elif result == 'loss':
        player_entry.losses += 1
    else:
        player_entry.draws += 1
    
    player_entry.score = player_entry.calculate_score()
    
    # Save updated leaderboard without score column
    with open('leaderboard.csv', 'w') as f:
        f.write("name,wins,losses,draws\n")  # Remove score from header
        for entry in entries:
            f.write(entry.to_csv_line())
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
