import json
import numpy as np
import math

word2pinyin = {}
word1 = {}
words2 = {}
la = 0.0001125

def read_word2pinyin():
    dict = {}
    with open('./word2pinyin.txt', 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split(" ")
            dict[words[0]] = words[1]
    return dict

def read_input(num):
    file_path = f'./{num}_word.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        json_file = json.load(file)
        for key in json_file:
            words = json_file[key]["words"]
            counts = json_file[key]["counts"]
            # 进行排序，将音-字对照表中的字按照出现次数排序
            sort = sorted({words[i]: counts[i] for i in range(len(words))}.items(), key=lambda x: x[1], reverse=True)
            json_file[key] = {pair[0]: pair[1] for pair in sort}
        return json_file
            
        
def viterbi(str: str):
    res = ""
    pinyins = str.strip().split(" ")
    layers = [None]
    props = []
    for pinyin in pinyins:
        layers.append([key for key in word1[pinyin]])
    for i in range(1, len(layers)):
        sum = np.sum(list(word1[word2pinyin[layers[i][0]]].values()))
        if layers[i-1] == None:
            props.append([(char, -math.log(word1[word2pinyin[char]][char] / sum) , '') for char in layers[i]])
        else:
            prop = []
            for char in layers[i]:
                (Q, finalPreId) = (math.inf, -1)
                for j, pre in enumerate(layers[i-1]):
                    prePinyin = word2pinyin[pre] + " " + word2pinyin[char]
                    preWord = pre + " " + char
                    condition = 0 # P(char | pre) = count(pre char) / count(pre)
                    curProp = (word1[word2pinyin[char]][char] / sum)
                    if prePinyin in words2 and preWord in words2[prePinyin]:
                        condition = words2[prePinyin][preWord] / word1[word2pinyin[pre]][pre]
                        # print(f"{prePinyin} : {preWord} : {words2[prePinyin][preWord]}")
                    condition = -math.log((1 - la) * condition + la * curProp)
                    # print(f"{pre} {char} : ({props[-1][j][0]}:{props[-1][j][1]}),  {condition},  {condition + props[-1][j][1]}")
                    if Q > condition + props[-1][j][1]:
                        Q = condition + props[-1][j][1]
                        finalPreId = j
                prop.append((char, Q, finalPreId))
            props.append(prop)
    choice = props[-1].index(min(props[-1], key=lambda x: x[1]))
    # print(props)
    for prop in props[::-1]:
        res += prop[choice][0]
        choice = prop[choice][2]
    return res[::-1]

if __name__ == '__main__':
    word2pinyin = read_word2pinyin()
    word1 = read_input(1)
    words2 = read_input(2)
    try:
        while True:
            line = input()
            print(viterbi(line))
    except EOFError:
        pass