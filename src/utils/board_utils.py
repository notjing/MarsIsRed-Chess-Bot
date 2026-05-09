import chess
import numpy as np

def get_mapped_coords(square, flip):
    """ Gets the accurate coordinates of a square depending on the POV """

    rank = chess.square_rank(square)
    file = chess.square_file(square)

    if flip:
        # to flip the board
        return rank, 7 - file
    else:
        return 7 - rank, file


def square_control(board):
    """ Returns 12 boards, each board indicating which squares are being attacked by each piece """

    flip = (board.turn == chess.BLACK)
    pieces_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    layers = []
    # loops through each colour and piece
    for colour in [board.turn, not board.turn]:
        for pt in pieces_types:

            # creates an nd array to represent the board
            grid = np.zeros((8, 8), dtype=np.float32)
            bit_attacking = 0

            # loops through each of the piece for a give type/colour
            for square in board.pieces(pt, colour):
                # returns a SquareSet of attacked sqrs (64 bit int)
                attacking = board.attacks(square)

                bit_board = int(attacking)
                bit_attacking |= bit_board

            # converts the SquareSet back into the ndarray (white pov)
            while bit_attacking > 0:
                lsb = bit_attacking & -bit_attacking
                r, c = get_mapped_coords(lsb.bit_length() - 1, flip)
                grid[r, c] = 1
                bit_attacking -= lsb

            layers.append(grid)

    return layers


def makeboards(board):
    """ Returns 12 boards, each with the position of the pieces """
    flip = (board.turn == chess.BLACK)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    layers = []

    # loops through each type / colour combo
    for colour in [board.turn, not board.turn]:
        for pt in piece_types:
            grid = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(pt, colour):
                r, c = get_mapped_coords(square, flip)
                grid[r][c] = 1
            layers.append(grid)

    return layers


def board_parameters(board):
    """ Gets all the basic numerical information from the board """

    # listicizes the castling rights
    rights = [
        board.has_kingside_castling_rights(board.turn),
        board.has_queenside_castling_rights(board.turn),
        board.has_kingside_castling_rights(not board.turn),
        board.has_queenside_castling_rights(not board.turn)
    ]

    # converts to 0/1
    castling = [1.0 if r else 0.0 for r in rights]

    counts = []

    # gets the counts of the pieces normalised to 0-1

    for colour in [board.turn,  not board.turn]:
        for pt in [[chess.PAWN, 8], [chess.KNIGHT, 2], [chess.BISHOP, 2], [chess.ROOK, 2], [chess.QUEEN, 1], [chess.KING, 1]]:
            counts.append(float(len(board.pieces(pt[0], colour)))/pt[1])

    # yea same here
    vals = {1: 1, 2: 3, 3: 3.4, 4: 5, 5: 9, 6: 0}
    score = 0
    for sq, pc in board.piece_map().items():
        v = vals[pc.piece_type]
        score += v if pc.color == board.turn else -v
    # emphasis more on material
    material_diff = [score / 10]

    in_check = [1.0 if board.is_check() else 0.0]
    is_mate = [1.0 if board.is_checkmate() else 0.0]

    return castling + counts + material_diff + in_check + is_mate
