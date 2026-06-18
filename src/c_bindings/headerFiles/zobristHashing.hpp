#include "chess.hpp"

typedef std::uint64_t u64;

struct Zobrist {
    u64 pieceSquare[12][64];
    u64 enPassant[8];
    u64 castlingRights[4];
    u64 sideToMove;

    Zobrist(); // Constructor declaration
};

extern Zobrist zobrist;

u64 updateZobristMove(u64 hsh, chess::Move mv, chess::Board b);
u64 generateHash(chess::Board board);
