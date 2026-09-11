#include "chess.hpp"
#include "headerFiles/zobristHashing.hpp"
#include <random>

typedef std::uint64_t u64;

Zobrist::Zobrist() {
    std::mt19937_64 rng(1070322);
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    for(int i = 0; i < 12; i++){
        for(int j = 0; j < 64; j++){
            pieceSquare[i][j] = dist(rng);
        }
    }

    for(int i = 0; i < 8; i++) enPassant[i] = dist(rng);
    for(int i = 0; i < 4; i++) castlingRights[i] = dist(rng);

    sideToMove = dist(rng);
}


Zobrist zobrist;

int pieceToIdx(chess::PieceType pt, chess::Color c){
   return int(pt) + 6 * (c == chess::Color::WHITE ? 0 : 1);
}

u64 handleZobristCastling(chess::Board board, u64 hash){
    std::string castleRights = board.getCastleString();

    bool whiteK = castleRights.find("K") != std::string::npos;
    bool whiteQ = castleRights.find("Q") != std::string::npos;
    bool blackK = castleRights.find("k") != std::string::npos;
    bool blackQ = castleRights.find("q") != std::string::npos;

    if (whiteK) hash ^= zobrist.castlingRights[0];
    if (whiteQ) hash ^= zobrist.castlingRights[1];
    if (blackK) hash ^= zobrist.castlingRights[2];
    if (blackQ) hash ^= zobrist.castlingRights[3];

    return hash;
}

u64 generateHash(chess::Board board){
    u64 hash = 0;

    // finding the pieces
    for(int idx = 0; idx < 64; idx++){
        chess::Piece p = board.at(chess::Square(idx));

        if(p == chess::Piece::NONE) continue;

        hash ^= zobrist.pieceSquare[pieceToIdx(p.type(), p.color())][idx];
    }

    // ep
    chess::Square epSquare = board.enpassantSq();

    if(epSquare != chess::Square::NO_SQ) hash ^= zobrist.enPassant[epSquare.file()];

    // castling
    hash = handleZobristCastling(board, hash);

    if(board.sideToMove() == chess::Color::BLACK) hash ^= zobrist.sideToMove;

    return hash;

}

u64 updateZobristMove (u64 hsh, chess::Move mv, chess::Board b){

    u64 newHsh = hsh;

    chess::Square fromSquare = mv.from();
    chess::Square toSquare = mv.to();

    chess::Piece movingPiece = b.at(fromSquare);

    // get rid of ep and castling
    if(b.enpassantSq() != chess::Square::NO_SQ){
        newHsh ^= zobrist.enPassant[b.enpassantSq().file()];
    }

    newHsh = handleZobristCastling(b, newHsh);


    // castling
    if (mv.typeOf() == chess::Move::CASTLING) {
        // Internal representation is king -> rook.
        const bool kingSide = toSquare > fromSquare;

        const auto kingTo =
            chess::Square::castling_king_square(kingSide, movingPiece.color());

        const auto rookFrom = toSquare;
        const auto rookTo =
            chess::Square::castling_rook_square(kingSide, movingPiece.color());

        // King: e1 -> g1, or e1 -> c1
        newHsh ^= zobrist.pieceSquare[
            pieceToIdx(movingPiece.type(), movingPiece.color())
        ][fromSquare.index()];

        newHsh ^= zobrist.pieceSquare[
            pieceToIdx(movingPiece.type(), movingPiece.color())
        ][kingTo.index()];

        // Rook: h1 -> f1, or a1 -> d1
        newHsh ^= zobrist.pieceSquare[
            pieceToIdx(chess::PieceType::ROOK, movingPiece.color())
        ][rookFrom.index()];

        newHsh ^= zobrist.pieceSquare[
            pieceToIdx(chess::PieceType::ROOK, movingPiece.color())
        ][rookTo.index()];
    }
    else{
        // moved piece
        newHsh ^= zobrist.pieceSquare[pieceToIdx(movingPiece.type(), movingPiece.color())][fromSquare.index()];
        newHsh ^= zobrist.pieceSquare[pieceToIdx(movingPiece.type(), movingPiece.color())][toSquare.index()];


        // captured piece
        if(b.isCapture(mv)){

            if(mv.typeOf() == chess::Move::ENPASSANT) {
                chess::Square capturedSquare = chess::Square(fromSquare.rank(), toSquare.file());
                chess::Piece capturedPiece = b.at(capturedSquare);

                newHsh ^= zobrist.pieceSquare[pieceToIdx(capturedPiece.type(), capturedPiece.color())][capturedSquare.index()];
            }
            else {
                chess::Piece capturedPiece = b.at(toSquare);
                newHsh ^= zobrist.pieceSquare[pieceToIdx(capturedPiece.type(), capturedPiece.color())][toSquare.index()];
            }
        }

        // promotion
        if(mv.typeOf() == chess::Move::PROMOTION){
            chess::PieceType pt = mv.promotionType();

            // gets rid of the pawn
            newHsh ^= zobrist.pieceSquare[pieceToIdx(movingPiece.type(), movingPiece.color())][toSquare.index()];
            newHsh ^= zobrist.pieceSquare[pieceToIdx(pt, movingPiece.color())][toSquare.index()];
        }
    }

    b.makeMove(mv);

    // put back ep and castling rights
    if(b.enpassantSq() != chess::Square::NO_SQ){
        newHsh ^= zobrist.enPassant[b.enpassantSq().file()];
    }

    newHsh = handleZobristCastling(b, newHsh);

    newHsh ^= zobrist.sideToMove;

    return newHsh;
}
