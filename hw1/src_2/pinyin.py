import json
import numpy as np
import math
import functools
import operator
import sys
import time

la = 0.00000001

def wrapper(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        print("-------------")
        print(f"Strat {fn.__name__}")
        var = fn(*args, **kwargs)
        print(f"End {fn.__name__}")
        print("-------------")
        return var
    return wrapped_fn

# @wrapper
def read_word2pinyin():
    with open('./word2pinyin.json', 'r', encoding='utf-8') as file:
        return json.load(file)

# @wrapper
def read_input(num):
    file_path = './pinyin2wordscounts.json' if num == 2 else './pinyin2charcount.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        json_file = json.load(file)
        for key in json_file:
            wordscounts = json_file[key]
            # 进行排序，将音-字对照表中的字按照出现次数排序
            json_file[key] = dict(sorted(wordscounts.items(), key=operator.itemgetter(1), reverse=True))
        # print(json_file)
    return json_file
            

def viterbi(str: str):
    res = ""
    pinyins = str.strip().split(" ")
    layers = [None]
    props = []
    for pinyin in pinyins:
        layers.append([key for key in word1[pinyin]])
    for i in range(1, len(layers)):
        sum = np.sum(list(word1[pinyins[i-1]].values()))
        if layers[i-1] == None:
            props.append([(char, -math.log(word1[pinyins[i-1]][char] / sum) , '') for char in layers[i]])
        else:
            prop = []
            for char in layers[i]:
                (Q, finalPreId) = (math.inf, -1)
                for j, pre in enumerate(layers[i-1]):
                    prePinyin = pinyins[i-2] + " " + pinyins[i-1]
                    preWord = pre + " " + char
                    condition = 0 # P(char | pre) = count(pre char) / count(pre)
                    curProp = (word1[pinyins[i-1]][char] / sum)
                    if prePinyin in words2 and preWord in words2[prePinyin]:
                        condition = words2[prePinyin][preWord] / word1[pinyins[i-2]][pre]
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
    argvs = sys.argv
    type = 1
    if len(argvs) >= 2:
        type = (int)(argvs[1])
    if type == 0:
        if not len(argvs) >= 4:
            print("You need to indicate the correct input and output file!")
            exit(1)
        inputfile = argvs[2]
        outputfile = argvs[3]
        stdoutputfile = '' if len(argvs) == 4 else argvs[4]
        if len(argvs) > 5:
            la = (float)(argvs[5])
        std_input = open(inputfile, 'r', encoding='utf-8')
        output = open(outputfile, 'a', encoding='utf-8')
        std_output = open(stdoutputfile, 'r', encoding='utf-8') if stdoutputfile != '' else None 
        output.truncate(0)
        std_input_pinyins = std_input.readlines()
        std_output_words = std_output.readlines() if std_output else None
        correct = 0
        times = []
        for i in range(len(std_input_pinyins)):
            start = time.time()
            output_words = viterbi(std_input_pinyins[i])
            end = time.time()
            times.append((end - start))
            output.write(output_words + '\n')
            if(std_output_words and output_words == std_output_words[i].strip()):
                correct += 1
        if std_output_words:
            print(f"Lambda: {la}\nAccuracy: ", correct/len(std_input_pinyins))
        mean = np.mean(times)
        sum = np.sum(times)
        print(f"Mean time: {mean} s.\nSum time: {sum} s.")
        print("Done!")
    else: 
        print("******************")
        print("Please input:")
        print("******************")
        try:
            while True:
                line = input()
                print(viterbi(line))
        except EOFError:
            pass