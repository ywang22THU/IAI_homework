import json
import numpy as np
import math
import functools
import operator
import sys
import time

la = 0.0001
eta = 0.1

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
    with open('./pinyin2word.json', 'r', encoding='utf-8') as file:
        return json.load(file)

# @wrapper
def read_input(num):
    file_path = './words3counts.json' \
                if num == 3 \
                else ('./words2counts.json' \
                      if num == 2 \
                      else './char2count.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        json_file = json.load(file)
            # 进行排序，将音-字对照表中的字按照出现次数排序
        json_file = dict(sorted(json_file.items(), key=operator.itemgetter(1), reverse=True))
        # print(json_file)
    return json_file
            

def viterbi(str: str):
    res = ""
    pinyins = str.strip().split(" ")
    layers = [None]
    props = []
    for pinyin in pinyins:
        layers.append(pinyin2word[pinyin])
    for i in range(1, len(layers)):
        sum = 1000000
        if layers[i-1] == None: # 首层 只考虑单字拼音
            props.append([(char, -math.log((word1[char] if char in word1 else 10) / sum) , '') for char in layers[i]])
        elif layers[i-2] == None: # 次层 只考虑双字拼音
            prop = []
            for char in layers[i]:
                (Q, finalPreId) = (math.inf, -1)
                for j, pre in enumerate(layers[i-1]):
                    word2 = pre + char
                    condition = 0 # P(char | pre) = count(pre char) / count(pre)
                    curProp = ((word1[char] if char in word1 else 10) / sum)
                    if pre in word1 and word2 in words2:
                        condition = words2[word2] / word1[pre]
                        # print(f"{prePinyin} : {preWord} : {words2[prePinyin][preWord]}")
                    condition = -math.log((1 - la) * condition + la * curProp)
                    # print(f"{pre} {char} : ({props[-1][j][0]}:{props[-1][j][1]}),  {condition},  {condition + props[-1][j][1]}")
                    if Q > condition + props[-1][j][1]:
                        Q = condition + props[-1][j][1]
                        finalPreId = j
                prop.append((char, Q, finalPreId))
            props.append(prop)
        else: # 其余层 考虑三字拼音
            prop = []
            for char in layers[i]:
                (Q, finalPreId) = (math.inf, -1)
                for j, pre in enumerate(layers[i-1]):
                    prepre = layers[i-2][props[-1][j][2]]
                    firstword2 = prepre + pre
                    secondword2 = pre + char
                    word3 = prepre + pre + char
                    condition2 = 0 # P(char | pre) = count(pre char) / count(pre)
                    condition3 = 0 # P(char | prepre pre) = count(prepre pre char) / count(prepre pre)
                    curProp = ((word1[char] if char in word1 else 10) / sum)
                    if pre in word1 and secondword2 in words2:
                        condition2 = words2[secondword2] / word1[pre]
                    if firstword2 in words2 and word3 in words3:
                        condition3 = words3[word3] / words2[firstword2]
                        # print(f"{prePinyin} : {preWord} : {words2[prePinyin][preWord]}")
                    condition = -math.log((1-eta) * condition3 + eta * ((1 - la) * condition2 + la * curProp))
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
    pinyin2word = read_word2pinyin()
    word1 = read_input(1)
    words2 = read_input(2)
    words3 = read_input(3)
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
        if len(argvs) > 6:
            la = (float)(argvs[5])
            eta = (float)(argvs[6])
        std_input = open(inputfile, 'r', encoding='utf-8')
        output = open(outputfile, 'a', encoding='utf-8')
        std_output = open(stdoutputfile, 'r', encoding='utf-8') if stdoutputfile != '' else None 
        output.truncate(0)
        std_input_pinyins = std_input.readlines()
        std_output_words = std_output.readlines() if std_output else None
        sencorrect = 0
        worcorrect = 0
        wholeword = 0
        times = []
        for i, input_word in enumerate(std_input_pinyins):
            start = time.time()
            output_words = viterbi(input_word)
            end = time.time()
            times.append((end - start))
            output.write(output_words + '\n')
            # print(f"Solving on {input_word}")
            if std_output_words:
                if output_words == std_output_words[i].strip():
                    sencorrect += 1
                for j, char in enumerate(output_words):
                    wholeword += 1
                    worcorrect += 1 if char == std_output_words[i].strip()[j] else 0
        if std_output_words:
            print(f"Lambda: {la} Eta: {eta}\nSentence Accuracy: ", sencorrect/len(std_input_pinyins), "\nWord Accuracy: ", worcorrect/wholeword)
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