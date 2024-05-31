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


pinyin2chars = {}
chars = []
char2count = {}
words2counts = {}
words3counts = {}

def extract_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]+')  # 匹配中文字符的正则表达式
    initresult = pattern.findall(text)  # 使用findall函数找到所有匹配的结果
    result = list(filter(lambda x: len(x) > 1, initresult)) # 过滤结果
    return result

@wrapper
def read_pinyin(floder):
    try:
        file = open(f'..{floder}/pinyin2word.txt', 'r', encoding='utf-8')
        for line in file:
            words = line.strip().split(' ')[1:]
            pinyin = line.strip().split(' ')[0]
            pinyin2chars[pinyin] = words
            chars.extend(words)
    except:
        print("Error in read word2pinyin")
@wrapper 
def read_words(filepath):
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            sentences = extract_chinese(line)
            for sentence in sentences:
                for i in range(len(sentence)):
                    char = sentence[i]
                    if chars is not None and char not in chars:
                        continue
                    char2count[char] = 1 if not char in char2count else char2count[char] + 1
                    if chars is not None and i < len(sentence) - 1:
                        if sentence[i + 1] not in chars:
                            continue
                        words2 = sentence[i:i + 2]
                        words2counts[words2] = 1 if not words2 in words2counts else words2counts[words2] + 1
                    if chars is not None and i < len(sentence) - 2:
                        if sentence[i + 2] not in chars:
                            continue
                        words3 = sentence[i:i + 3]
                        words3counts[words3] = 1 if not words3 in words3counts else words3counts[words3] + 1
                        
                    

@wrapper
def write_output(filepath, dic: dict):
    dic = dict(filter(lambda x: not (isinstance(x[1], int) and x[1] < 10), dic.items()))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False)

def main():
    if len(sys.argv) != 2 or os.path.isdir(sys.argv[1]):
        print("You need to indicate the raw data floder")
        exit(1)
    pinyinOutputFile = './pinyin2word.json'
    charOutputFilePath = './char2counts.json'
    words2OutputFilePath = './words2counts.json'
    words3OutputFilePath = './words3counts.json'
    print("********************\n")
    print("Begin reading pinyins!\n")
    print("********************\n")
    read_pinyin(sys.argv[1])
    write_output(pinyinOutputFile, pinyin2chars)
    print("********************\n")
    print("Begin reading words!\n")
    print("********************\n")
    for file in os.listdir('..' + sys.argv[1]):
        inputFilePath = os.path.join(('..' + sys.argv[1]), file)
        if os.path.isfile(inputFilePath) and not 'pinyin2word.txt' in os.path.basename(file):            
            read_words(inputFilePath)
        print(f"{inputFilePath} Done!")
    write_output(charOutputFilePath, char2count)
    write_output(words2OutputFilePath, words2counts)
    write_output(words3OutputFilePath, words3counts)
    
if __name__ == "__main__":
    main()