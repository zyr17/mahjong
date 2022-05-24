#ifndef ROUNDTOWIN_H
#define ROUNDTOWIN_H

#include "Tile.h"
#include <map>

namespace_mahjong

class Syanten {
    const static int TILENUM = 38, INT_MAX = 2147483647;
    const static int SYANTENSYS = 4, TILESYS = 3;
    const int TNUM_TO_SNUM[34] = {
        0, 1, 2, 3, 4, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36
    };  // code in syanten calculation has different tile encoding
    std::unordered_map<int, int> syanten_map;
    int Td[5], Ttvec[5];
    std::vector<int> hand_to_tile_vec(const std::vector<Tile*> &hand);
    int chitoi_syanten(const std::vector<int> &bu);
    int kokushi_syanten(const std::vector<int> &bu);
    int calc_mentsu(const std::vector<int> &bu, int mentsu);
    int calc_syanten(const std::vector<Tile*> &hand, int fuuro, bool chitoikokushi = true);
    void load_syanten_map();
    bool is_loaded = false;
    Syanten() = default;
    Syanten (Syanten const &);
    Syanten& operator = (const Syanten &);
public:
    static Syanten& instance() {
        static Syanten inst;
        return inst;
    }
    int normal_round_to_win(const std::vector<Tile*>& hand, int num_副露);
};

namespace_mahjong_end
#endif // end #ifndef ROUNDTOWIN_H