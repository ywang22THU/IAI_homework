#ifndef MCT_H
#define MCT_H

#include<bits/stdc++.h>
#include"Point.h"
using namespace std;

const int RANDOM_RANGE[4][13] = {
    {0, 1, 3, 6, 10, 15, 19, 22, 24, 25, 0, 0, 0},
    {0, 1, 3, 6, 10, 15, 20, 24, 27, 29, 30, 0, 0},
    {0, 1, 3, 6, 10, 15, 21, 26, 30, 33, 35, 36, 0},
    {0, 1, 3, 6, 10, 15, 21, 27, 32, 36, 39, 41, 42}
};

int mid_rand(int N);
pair<Point, int> must(int M, int N, const int * top, int ** board);

class Node{               // 蒙特卡洛树的节点
    Node * parent;        // 父节点
    Node ** childs;       // 子节点
    int M, N;             // 棋盘行数与列数 
    int newX, newY;       // 相比于父节点新增的点，避免了每一个点都存一遍全局信息
    int owner;            // 该格的拥有者
    int winner;           // 该格对应棋局的胜者
    double UCB;           // 节点信心上限
    double sumProfit;     // 总收益，用于计算UCB
    double times;         // 访问次数，用于计算UCB
    int expables;         // 尚未被拓展的数量
    int expnum;           // 初始状态下能被拓展的数量
    bool * expd;          // 已经被拓展的列
    int * top;           // 每列最顶层
public:
    friend class MCT;
    Node();
    Node(int M, int N, int newX, int newY, int noX, int noY, const int * top);
    Node(Node * parent, int M, int N, int newX, int newY, int noX, int noY, int owner);
    ~Node();
};

class MCT{ // 蒙特卡洛树
    Node * root; // 树的根节点
    int M, N; // 棋盘大小
    int noX, noY; // 禁止点
    int ** initBoard;
    int ** board; // 初始棋局
public:
    MCT();
    MCT(int _M, int _N, int _newX, int _newY, int _noX, int _noY,  const int * _top, int ** _board);
    ~MCT();
    double UCBcacl(Node *); // 计算一个节点的UCB
    Node * choose(); // 选择节点
    Node * bestChild(Node *); // 选择UCB最大的孩子
    Node * winChild(); // 选择胜率最大的节点
    Node * expand(Node *); // 扩展节点
    pair<double, int> simulate(Node *); // 模拟节点
    void back(Node *, pair<double, int>); // 回传
    Point search(); // 蒙特卡洛搜索
};

#endif