import chess
import random

# -----------------------------
# Zobrist Hash Setup
# -----------------------------
ZOBRIST_PIECE = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]
ZOBRIST_BLACK_TO_MOVE = random.getrandbits(64)

# Optional: castling and en passant
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(16)]
ZOBRIST_EN_PASSANT = [random.getrandbits(64) for _ in range(8)]
def piece_index(piece):
    base = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }[piece.piece_type]
    return base + (0 if piece.color == chess.WHITE else 6)

def piece_value(piece):
    if not piece:
        return 0
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    return values[piece.piece_type]

TT = {}

class TTEntry:
    def __init__(self, depth, eval, flag, move):
        self.depth = depth
        self.eval = eval
        self.flag = flag  # "EXACT", "LOWERBOUND", "UPPERBOUND"
        self.move = move


def compute_zobrist(board):
    h = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = piece_index(piece)
            h ^= ZOBRIST_PIECE[idx][sq]
    if board.turn == chess.BLACK:
        h ^= ZOBRIST_BLACK_TO_MOVE

    # Castling rights
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    h ^= ZOBRIST_CASTLING[castling]

    # En passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        h ^= ZOBRIST_EN_PASSANT[file]

    return h


def compute_child_hash(board, move, current_hash):
    new_hash = current_hash

    from_sq = move.from_square
    to_sq = move.to_square
    piece = board.piece_at(from_sq)

    # deal with all old states

    # castling
    old_castling = 0
    if board.has_kingside_castling_rights(chess.WHITE): old_castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE): old_castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK): old_castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK): old_castling |= 8
    new_hash ^= ZOBRIST_CASTLING[old_castling]

    # enpassant Square
    if board.ep_square is not None:
        new_hash ^= ZOBRIST_EN_PASSANT[chess.square_file(board.ep_square)]

    # remove moving piece
    new_hash ^= ZOBRIST_PIECE[piece_index(piece)][from_sq]

    if board.is_en_passant(move):
        # remove enpassant captured pawn
        diff = -8 if piece.color == chess.WHITE else 8
        ep_captured_sq = to_sq + diff
        new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.PAWN, not piece.color))][ep_captured_sq]
    else:
        # normal capture
        captured_piece = board.piece_at(to_sq)
        if captured_piece:
            new_hash ^= ZOBRIST_PIECE[piece_index(captured_piece)][to_sq]

    # place moved/promoted piece
    if move.promotion:
        promo_piece = chess.Piece(move.promotion, piece.color)
        new_hash ^= ZOBRIST_PIECE[piece_index(promo_piece)][to_sq]
    else:
        new_hash ^= ZOBRIST_PIECE[piece_index(piece)][to_sq]

    # castling (rook)
    if board.is_castling(move):
        if to_sq == chess.G1:  # WK
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.WHITE))][chess.H1]
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.WHITE))][chess.F1]
        elif to_sq == chess.C1:  # WQ
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.WHITE))][chess.A1]
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.WHITE))][chess.D1]
        elif to_sq == chess.G8:  # BK
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.BLACK))][chess.H8]
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.BLACK))][chess.F8]
        elif to_sq == chess.C8:  # BQ
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.BLACK))][chess.A8]
            new_hash ^= ZOBRIST_PIECE[piece_index(chess.Piece(chess.ROOK, chess.BLACK))][chess.D8]

    # all new board state
    board.push(move)

    # castling
    new_castling = 0
    if board.has_kingside_castling_rights(chess.WHITE): new_castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE): new_castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK): new_castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK): new_castling |= 8
    new_hash ^= ZOBRIST_CASTLING[new_castling]

    # enpassant square
    if board.ep_square is not None:
        new_hash ^= ZOBRIST_EN_PASSANT[chess.square_file(board.ep_square)]

    new_hash ^= ZOBRIST_BLACK_TO_MOVE

    board.pop()

    return new_hash
