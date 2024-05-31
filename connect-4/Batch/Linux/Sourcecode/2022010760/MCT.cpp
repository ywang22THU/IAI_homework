#include "MCT.h"
#include "Judge.h"
#include "Point.h"
#include<bits/stdc++.h>
using namespace std;

const double C = 0.7;

int mid_rand(int N){ // 选择更居中的随机数
    const int * randomRange = RANDOM_RANGE[N - 9];
    int mod = randomRange[N];
    srand(time(0));
    int initRan = rand() % mod;
    for(int i = 0; i < N; i++){
        if(randomRange[i] <= initRan && initRan < randomRange[i + 1]){
            return i;
        }
    }
    return -1;
}

pair<Point, int> must(int M, int N, const int * top, int ** board){
    // 判断有无直接胜利的走法
    for(int y = 0; y < N; y++){
        int x = top[y];
        if(x < 0)
            continue;
        board[x][y] = 2;
        if(machineWin(x, y, M, N, board)){
            board[x][y] = 0;
            return {Point(x, y), 2};
        }
        board[x][y] = 0;
    }
    // 判断有无非走即输的走法
    for(int y = 0; y < N; y++){
        int x = top[y];
        if(x < 0)
            continue;
        board[x][y] = 1;
        if(userWin(x, y, M, N, board)){
            board[x][y] = 0;
            return {Point(x, y), 1};
        }
        board[x][y] = 0;
    }
    return {Point(-1, -1), -1};
}

Node::Node(){}

Node::Node(int M, int N, int newX, int newY, int noX, int noY, const int * _top){
    this->parent = nullptr;
    this->M = M;
    this->N = N;
    this->newX = newX;
    this->newY = newY;
    this->owner = 1; // 这个构造函数用于构造树根，因此其一定是对手下的
    this->winner = -1;
    this->expables = 0;
    this->times = 0.0;
    this->sumProfit = 0.0;
    this->top = new int[N];
    this->expd = new bool[N];
    this->childs = new Node*[N];
    this->UCB = 0.0;
    for(int y = 0; y < N; y++){
        childs[y] = nullptr;
        this->top[y] = _top[y] - 1;
        if(this->top[y] == noX && y == noY){
            this->top[y] -= 1;
        }
        this->expd[y] = this->top[y] < 0;
        this->expables += (this->top[y] >= 0) ? 1 : 0;
    }
    this->expnum = this->expables;
}

Node::Node(Node * parent, int M, int N, int newX, int newY, int noX, int noY, int owner){
    this->parent = parent;
    this->M = M;
    this->N = N;
    this->newX = newX;
    this->newY = newY;
    this->owner = owner;
    this->winner = -1;
    this->sumProfit = 0.0;
    this->times = 1.0;
    this->expables = 0;
    this->top = new int[N];
    this->expd = new bool[N];
    this->childs = new Node*[N];
    this->UCB = 0.0;
    for(int y = 0; y < N; y++){
        childs[y] = nullptr;
        this->top[y] = parent->top[y] - (y == newY ? 1 : 0);
        if(this->top[y] == noX && y == noY){
            this->top[y] -= 1;
        }
        this->expables += (this->top[y] >= 0 ? 1 : 0);
        this->expd[y] = this->top[y] < 0;
    }
    this->expnum = this->expables;
}

Node::~Node(){
    delete[] top;
    delete[] expd;
    for(int y = 0; y < N; y++){
        if(childs[y] != nullptr){
            delete childs[y];
        }
    }
}

double MCT::UCBcacl(Node * node){
    double average = node->sumProfit / (node->times + 1e-5);
    double explore = C * sqrt(2 * (log(this->root->times) / (node->times + 1e-5)));
    node->UCB = average + explore;
    return node->UCB;
}

MCT::MCT(){}

MCT::MCT(int _M, int _N, int _newX, int _newY, int _noX, int _noY, const int * _top, int ** _board){
    this->M = _M;
    this->N = _N;
    this->noX = _noX;
    this->noY = _noY;
    this->root = new Node(_M, _N, _newX, _newY, _noX, _noY, _top);
    this->initBoard = new int *[_M];
    this->board = new int*[_M];
    for(int x = 0; x < _M; x++){
        this->initBoard[x] = new int[_N];
        this->board[x] = new int[_N];
        for(int y = 0; y < _N; y++){    
            this->initBoard[x][y] = _board[x][y];
            this->board[x][y] = _board[x][y];
        }
    }
}

MCT::~MCT(){
    for(int x = 0; x < this->M; x++){
        delete[] this->board[x];
        delete[] this->initBoard[x];
    }
    delete root;
}

Node * MCT::winChild(){
    double win = -1000000000.0;
    int winId = -1;
    for(int y = 0; y < root->N; y++){
        if(root->childs[y] != nullptr){
            if(root->childs[y]->sumProfit / root->childs[y]->times > win){
                win = root->childs[y]->sumProfit / root->childs[y]->times;
                winId = y;
            }
        }
    }
    return (winId == -1) ? nullptr : root->childs[winId];
}

Node * MCT::bestChild(Node * node){
    double maxUCB = -1000000000.0;
    int maxId = -1;
    for(int y = 0; y < node->N; y++){ // 遍历所有孩子，找到其中UCB最大的那个
        if(node->childs[y] != nullptr && node->childs[y]->UCB > maxUCB){
            maxUCB = node->childs[y]->UCB;
            maxId = y;
        }
    }
    return (maxId == -1) ? nullptr : node->childs[maxId];
}

Node * MCT::choose(){
    Node * res = this->root;
    for(int x = 0; x < this->M; x++){ // 每次开始选择的时候，复位整个棋盘
        for(int y = 0; y < this->N; y++){
            this->board[x][y] = this->initBoard[x][y];
        }
    }
    while(true){
        res->times += 1;
        if(res->expables <= 0){ // 如果不可扩展，则选择UCB值最大的子节点，继续循环
            if(res->winner != -1){
                return res;
            }
            res = bestChild(res); // 选择最优节点
            this->board[res->newX][res->newY] = res->owner; // 更新棋盘
        }
        else{ // 如果可以扩展，则扩展
            return this->expand(res);
        }
    }
}

Node * MCT::expand(Node * node){ // 扩展节点
    int expY = rand() % N;
    while(node->expd[expY]){ // 随机找到一个可扩展点
        expY = (expY + 1) % N;
    }
    int expX = node->top[expY]; // 根据列找到行坐标
    int owner = 3 - node->owner; // 确定这个节点的拥有者，即与其父节点相反
    Node * expNode = new Node(node, this->M, this->N, expX, expY, this->noX, this->noY, owner);
    node->expd[expY] = true;
    node->expables -= 1;
    node->childs[expY] = expNode;
    this->board[expX][expY] = owner;
    if(owner == 1 && userWin(expX, expY, this->M, this->N, this->board)){ // 如果是对方下棋并且获胜
        expNode->winner = 1;
        expNode->sumProfit = 1.0;
        expNode->expables = -1;
        expNode->expnum = -1;
    }
    if(owner == 2 && machineWin(expX, expY, this->M, this->N, this->board)){ // 如果是我方下棋并且获胜
        expNode->winner = 2;
        expNode->sumProfit = 1.0;
        expNode->expables = -1;
        expNode->expnum = -1;
    }
    if(isTie(this->N, expNode->top)){ // 如果是平局
        expNode->winner = 0;
        expNode->sumProfit = -1.0;
        expNode->expables = -1;
        expNode->expnum = -1;
    }
    return expNode; 
}

pair<double, int> MCT::simulate(Node * node){ // 模拟节点
    int count = 0;
    if(node->winner == 0){
        return {0.0, 0};
    }
    if(node->winner != -1){
        return {node->winner == node->owner ? 1.0 : -2.0, 0};
    }
    int player = 3 - node->owner;
    bool running = node->expables > 0;
    int * simtop = new int [node->N];
    for(int y = 0; y < node->N; y ++){
        simtop[y] = node->top[y];
    }
    int simexpables = node->expnum;
    int newX, newY = 0;
    pair<Point, int> mustStep = {Point(-1, -1), -1};
    while(running){
        mustStep = must(this->M, this->N, simtop, this->board);
        if(mustStep.second != -1){
            newX = mustStep.first.x;
            newY = mustStep.first.y;
        }
        else{
            count += 1;
            newY = rand() % N;
            int dic = rand() % 2 ? 1 : -1;
            while(simtop[newY] < 0){ // 随机找到一个可落子点
                newY = (newY + dic + N) % (this->N);
            }
        }
        newX = simtop[newY];
        player = 3 - player;
        this->board[newX][newY] = player;
        if((player == 1 && userWin(newX, newY, this->M, this->N, this->board))||
            (player == 2 && machineWin(newX, newY, this->M, this->N, this->board))){
            delete[] simtop; 
            return {node->owner == player ? 1.0 : -1.0, count};
        }
        simtop[newY] -= 1;
        if(simtop[newY] == this->noX && newY == noY){
            simtop[newY] -= 1;
        }
        if(simtop[newY] < 0){
            simexpables -= 1;
        }
        running = simexpables > 0;
    }
    delete[] simtop;
    return {-1.0, count}; // 如果最终不可扩展了，则一定是平局了
}

void MCT::back(Node * node, pair<double, int> sim){ // 回传，模拟的收益为sim.first，次数为sim.second
    double backVals = sim.first;
    double backValn = sim.first == -1.0 ? -1.0 : -1.0 - sim.first;
    for(int i = 0; i < sim.second; i += 15){
        backVals *= 0.9;
        backValn *= 0.9;
    }
    node->sumProfit += backVals;
    this->UCBcacl(node);
    Node * par = node->parent;
    while(par != nullptr){
        par->sumProfit += (par->owner == node->owner) ? backVals : backValn;
        this->UCBcacl(par);
        par = par->parent;
    }
}

Point MCT::search(){
    clock_t time_s = clock();
	pair<Point, int> mustStep = must(M, N, this->root->top, this->board);
    if(mustStep.second != -1){
        return mustStep.first;
    }
    Node * chooseNode =  nullptr;
    while(true){
        if((double)(clock() - time_s) / CLOCKS_PER_SEC > 2.4){
            break;
        }
        chooseNode = this->choose();
        pair<double, int> sim = this->simulate(chooseNode);
        this->back(chooseNode, sim);
    }
    Node * nextStep = this->winChild();
    Point res = Point(nextStep->newX, nextStep->newY);
    return res;
}