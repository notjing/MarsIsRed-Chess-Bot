from ChessAI.src.old_search.utils import compute_child_hash, piece_value, TT

def score_move(board, move, prev_move, parent_hash=None):
    score = 0

    # PV move first
    if prev_move is not None and move == prev_move:
        return 1000000

    # Check bonus
    board.push(move)
    if board.is_check():
        score += 1000

    #prioritize castling
    if board.is_castling(move):
        score += 500
    board.pop()

    # Captures (MVV-LVA)
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        attacker_value = piece_value(board.piece_at(move.from_square))
        victim_value = piece_value(captured)
        score += 1000 + victim_value - attacker_value

    # Promotions
    if move.promotion:
        score += 900

    if parent_hash is not None:
        child_hash = compute_child_hash(board, move, parent_hash)
        entry = TT.get(child_hash)

        if entry:
            # exact means that the search involving this move was completely resolved; true score known
            if entry.flag == "EXACT":
                score += 3000 + entry.eval
            # the move was good enough to cause a "beta cutoff" meaning that the score is at least this value
            elif entry.flag == "LOWERBOUND":  # likely good
                score += 2000
            elif entry.flag == "UPPERBOUND":  # weaker info
                score += 1000

    return score
