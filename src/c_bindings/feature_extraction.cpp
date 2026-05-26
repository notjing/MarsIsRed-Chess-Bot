#include "chess.hpp"
#include "feature_extraction.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

chess::Square getMappedSquare(chess::Square sqr, bool flip) {
    int row = sqr.rank();
    int col = sqr.file();

    if (flip) {
        return chess::Square(static_cast<chess::Rank>(row), static_cast<chess::File>(col));
    } else {
        return chess::Square(static_cast<chess::Rank>(7 - row), static_cast<chess::File>(col));
    }
}
std::vector<chess::Square> getSquaresFromBitboard(chess::Bitboard bB){
    std::vector<chess::Square> squares;
    while (bB) {
        squares.push_back(bB.pop());
    }

    return squares;
}

chess::Bitboard getAttackingFromSqr(chess::Square sqr, chess::PieceType pt, chess::Color c, chess::Bitboard occ){

    if(pt == chess::PieceType::PAWN) return chess::attacks::pawn(c, sqr);
    if(pt == chess::PieceType::KNIGHT) return chess::attacks::knight(sqr);
    if(pt == chess::PieceType::BISHOP) return chess::attacks::bishop(sqr, occ);
    if(pt == chess::PieceType::ROOK) return chess::attacks::rook(sqr, occ);
    if(pt == chess::PieceType::QUEEN) return chess::attacks::queen(sqr, occ);
    if(pt == chess::PieceType::KING) return chess::attacks::king(sqr);

    return chess::Bitboard(0);

}

py::array_t<float> boardParams(chess::Board board){
    bool flip = board.sideToMove() == chess::Color::BLACK;
    chess::PieceType pieceTypes[6] = {chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP, chess::PieceType::ROOK, chess::PieceType::QUEEN, chess::PieceType::KING};
    chess::Color colours[2] = {board.sideToMove(), board.sideToMove() == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE};

    py::array_t<float> layers({8,8,25});
    auto layersMut = layers.mutable_unchecked<3>();
    std::fill(layers.mutable_data(), layers.mutable_data() + layers.size(), 0.0f);

    int idx = 0;

    for(auto colour : colours){
        for(auto pt : pieceTypes){

            float attackGrid[8][8] = {};
            float positionGrid[8][8] = {};
            chess::Bitboard bitAttacking;

            std::vector<chess::Square> sqrs = getSquaresFromBitboard(board.pieces(pt, colour));

            for(int i = 0; i < sqrs.size(); i++){
                chess::Square sqr = getMappedSquare(sqrs[i], flip);
                chess::Bitboard attacking = getAttackingFromSqr(sqrs[i], pt, colour, board.occ());                bitAttacking |= attacking.getBits();
                positionGrid[sqr.rank()][sqr.file()] = 1;
            }

            while (bitAttacking) {
                chess::Square attackSqr = bitAttacking.pop();
                chess::Square mappedAttack = getMappedSquare(attackSqr, flip);
                attackGrid[mappedAttack.rank()][mappedAttack.file()] = 1;
            }

            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 8; j++){
                    layersMut(i ,j, idx) = positionGrid[i][j];
                    layersMut(i ,j, idx + 12) = attackGrid[i][j];
                }
            }

            idx++;

        }
    }

    chess::Square epSquare = board.enpassantSq();
    int enPassantGrid[8][8] = {};

    if(epSquare != chess::Square::underlying::NO_SQ){
        epSquare = getMappedSquare(epSquare, flip);
        enPassantGrid[epSquare.rank()][epSquare.file()] = 1;
    }

    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            layersMut(i ,j, 24) = enPassantGrid[i][j];
        }
    }

    return layers;

}

py::array_t<float> denseParams(chess::Board board){

    py::array_t<float> params({19});
    auto paramsMut = params.mutable_unchecked<1>();
    std::fill(params.mutable_data(), params.mutable_data() + params.size(), 0.0f);

    bool flip = board.sideToMove() == chess::Color::BLACK;
    chess::PieceType pieceTypes[6] = {chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP, chess::PieceType::ROOK, chess::PieceType::QUEEN, chess::PieceType::KING};
    float pieceCounts[6] = {8.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f};
    float pieceVal[6] = {1.0f, 3.0f, 3.4f, 5.0f, 9.0f, 0.0f};
    chess::Color colours[2] = {board.sideToMove(), board.sideToMove() == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE};
    int idx = 4;

    std::string castleRights = board.getCastleString();

    bool whiteK = castleRights.find("K") != std::string::npos;
    bool whiteQ = castleRights.find("Q") != std::string::npos;
    bool blackK = castleRights.find("k") != std::string::npos;
    bool blackQ = castleRights.find("q") != std::string::npos;

    paramsMut(0) = (flip ? blackK : whiteK) ? 1.0f : 0.0f;
    paramsMut(1) = (flip ? blackQ : whiteQ) ? 1.0f : 0.0f;
    paramsMut(2) = (!flip ? blackK : whiteK) ? 1.0f : 0.0f;
    paramsMut(3) = (!flip ? blackQ : whiteQ) ? 1.0f : 0.0f;

    for(auto colour : colours){
        int j = 0;
        for(auto pt : pieceTypes){
            paramsMut(idx) = board.pieces(pt, colour).count() / pieceCounts[j];
            idx++;
            j++;
        }
    }

    float matDiff = 0;

    for(auto colour : colours){
        int j = 0;
        for(auto pt : pieceTypes){
            matDiff += colour == board.sideToMove() ? board.pieces(pt, colour).count() * pieceVal[j] : -board.pieces(pt, colour).count() * pieceVal[j];
            j++;
        }
    }

    paramsMut(idx) = matDiff / 10.0f;
    idx++;

    paramsMut(idx) = board.inCheck() ? 1.0f : 0.0f;
    idx++;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty() && board.inCheck()) paramsMut(idx) = 1.0f;
    else paramsMut(idx) = 0.0f;


    return params;
}

