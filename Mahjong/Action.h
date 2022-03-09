#ifndef ACTION_H
#define ACTION_H

#include "Tile.h"
#include <tuple>

enum class Action : uint8_t {
	// response begin
	pass,
	吃, 
	碰,
	杠,
	荣和,
	// response end
	// 注意到所有的response Action可以通过大小来比较

	抢暗杠,
	抢杠,

	// self action begin
	暗杠,
	加杠,
	出牌,
	立直,
	自摸,
	九种九牌,
	// self action end
};

struct SelfAction {
	SelfAction();
	Action action;
	std::vector<Tile*> correspond_tiles;
	SelfAction(Action, std::vector<Tile*>);
	inline bool operator<(const SelfAction& other) const
	{
		return tie(action, correspond_tiles) < tie(action, correspond_tiles);
	}
	std::string to_string() const;
};

struct ResponseAction {
	ResponseAction();
	Action action;
	std::string to_string() const;
	std::vector<Tile*> correspond_tiles;
	inline bool operator<(const ResponseAction& other) const
	{
		return tie(action, correspond_tiles) < tie(action, correspond_tiles);
	}
};

#endif
