import re
import json
import functools
import sys
import os

def wrapper(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        print("-------------")
        print(f"Start {fn.__name__}")
        fn(*args, **kwargs)
        print(f"End {fn.__name__}")
        print("-------------")
    return wrapped_fn


word2pinyin = {}
pinyin2charcount = {}
pinyin2wordscounts = {}

def extract_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]+')  # 匹配中文字符的正则表达式
    initresult = pattern.findall(text)  # 使用findall函数找到所有匹配的结果
    result = list(filter(lambda x: len(x) > 1, initresult)) # 过滤结果
    return result

@wrapper
def read_word2pinyin(floder):
    file = open(f'..{floder}/pinyin2word.txt', 'r', encoding='utf-8')
    for line in file:
        words = line.strip().split(' ')[1:]
        pinyin = line.strip().split(' ')[0]
        for word in words:
            if word in word2pinyin:
                word2pinyin[word].append(pinyin)
            else:
                word2pinyin[word] = [pinyin]
            
    
@wrapper
def read_char(filepath):
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            sentences = extract_chinese(line)
            for sentence in sentences:
                for i in range(len(sentence)):
                    if sentence[i] not in word2pinyin:
                        continue
                    pinyins = word2pinyin[sentence[i]]
                    word = sentence[i]
                    for pinyin in pinyins:
                        if pinyin in pinyin2charcount:
                            if word in pinyin2charcount[pinyin]:
                                pinyin2charcount[pinyin][word] += 1
                            else:
                                pinyin2charcount[pinyin][word] = 1
                        else:
                            pinyin2charcount[pinyin] = {word: 1} 
    
@wrapper 
def read_words(filepath):
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            sentences = extract_chinese(line)
            for sentence in sentences:
                for i in range(len(sentence) - 1):
                    if sentence[i] not in word2pinyin or sentence[i+1] not in word2pinyin:
                        continue
                    pinyins = []
                    for first in word2pinyin[sentence[i]]:
                        for second in word2pinyin[sentence[i+1]]:
                            pinyins.append(f"{first} {second}")
                    word = sentence[i] + " " + sentence[i+1]
                    for pinyin in pinyins:
                        if pinyin in pinyin2wordscounts:
                            if word in pinyin2wordscounts[pinyin]:
                                pinyin2wordscounts[pinyin][word] += 1
                            else:
                                pinyin2wordscounts[pinyin][word] = 1
                        else:
                            pinyin2wordscounts[pinyin] = {word: 1}

@wrapper
def write_output(filepath, dic: dict):
    for key in dic:
        if isinstance(dic[key], int) and dic[key] < 5:
            dic.pop(key)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False)

def main():
    if len(sys.argv) != 2 or os.path.isdir(sys.argv[1]):
        print("You need to indicate the raw data floder")
        exit(1)
    pinyinOutputFile = './word2pinyin.json'
    charOutputFilePath = './pinyin2charcount.json'
    wordsOutputFilePath = './pinyin2wordscounts.json'
    print("********************\n")
    print("Begin reading pinyins!\n")
    print("********************\n")
    read_word2pinyin(sys.argv[1])
    write_output(pinyinOutputFile, word2pinyin)
    print("********************\n")
    print("Begin reading chars and words!\n")
    print("********************\n")
    for file in os.listdir('..' + sys.argv[1]):
        inputFilePath = os.path.join(('..' + sys.argv[1]), file)
        if os.path.isfile(inputFilePath) and not 'pinyin2word.txt' in os.path.basename(file):            
            read_char(inputFilePath)
            read_words(inputFilePath)
        print(f"{inputFilePath} Done!")
    write_output(charOutputFilePath, pinyin2charcount)
    write_output(wordsOutputFilePath, pinyin2wordscounts)
    
if __name__ == "__main__":
    main()