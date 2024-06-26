# 拼音输入法 实验报告

## 实验环境
```python
python 3.10.13
numpy 1.26.4
```

## 语料处理
使用的语料库为下发的新浪语料与微博语料，二者共同处理得到数据，预处理方法如下：
- 首先生成字-音表：按行读入音-字表，并将每个汉字的读音存成如下的```json```字典格式：  
  
    ```
    <word>: [pinyin1, pinyin2, ...]
    ```
    多音字的```value```需要是一个列表，为了处理方便，将所有字对应的值都设置成列表，后续处理中只需要遍历列表即可，省去了特判
- 然后得到字频：按行读入，利用正则表达式提取每一行中的中文字符，将字频存成如下的```json```格式：
    ```
    <pinyin>: [
        <char1>: <count1>,
        <char2>: <count2>,
        ...
    ]
    ```
    对于多音字，其会出现在每一个读音中，并且对应的数量是相同的，也即不会区分原文本中的正确读音是哪一种
- 最后处理词频：按行读入，利用正则表达式提取每一行中的连续中文字符，丢弃所有的单字，将词频存成如下的```json```格式：
    ```
    <pinyin>: [
        <word1>: <count1>,
        <word2>: <count2>,
        ...
    ]
    ```
- 在三元模型中，因为这样处理所需要的空间过大，因此对```json```进行扁平化处理，得到的结果均为一层的字典，并且删掉了所有出现频率小于$10$的字/词
## 基于字的二元模型输入法

### 基本思路
使用$\text{Viterbi}$算法计算对于输入的每一个前缀，各种组合的不同可能性，同时利用在条件概率中添加单字概率的方法来处理未出现二元组的情况

### 公式推导
假设输入为拼音串$O$，输出为字符串$S = w_1w_2\dots w_n$，则我们应该让概率$P(S | O)$尽可能大，由条件概率公式：
$$P(S | O) = \frac{P(O | S)P(S)}{P(O)}$$
其中：$P(O)$为常量，$P(O | S)\approx 1$，因此只需要让$P(S)$概率最大即可  
由于我们考虑的是二元模型，因此每个字的出现只与其前方一个字有关，有：
$$P(S) = \prod\limits_{i=1}^{n}P(w_i  | w_{i-1})$$
为计算简便可取负对数，因此只用使得下面的式子尽量大
$$-\sum\limits_{i=1}^{n}\log(P(w_{i} |  w_{i-1}))$$
而$P(w_{i} | w_{i-1}) = P(w_{i-1}w_{i})/P(w_{i})$，因此如果$w_{i-1}w_{i}$这个字串在语料库中从未出现过，会导致概率为$0$，出现对$0$取对数的操作，为了避免，采用如下的平滑化方式
$$(1-\lambda)P(w_{i} | w_{i-1}) + \lambda P(w_i)\rightarrow P(w_{i} | w_{i-1})$$
这样每一个字串都被赋予了一个最低值概率，其中的$\lambda$为可调参数  

### 实现过程
注：以下所述的所有概率均经过取负对数处理
- 读入数据，将与处理过后的语料库存在字典中
- 读入输入，并且将输入按照空格分割，对于每一个拼音，将其所对应的所有可能的字加入栅栏图的同一层，得到完整的栅栏图
- 对于栅栏图的首层，其出现概率直接等于每一个字在该拼音下出现的概率
- 对于从第二层开始之后的每一层，其中任何一个字出现的概率为他在前一层的条件下出现概率的最大值，转移方程如下：
$$P(w_{i}) = \max (P(w_{i-1}) + P(w_{i} | w_{i-1}))\quad\forall w_{i-1}\in \text{layers}[i-1]$$
- 对于处理后的每一层，其中应当包含每一个字出现的最大概率，以及这个概率对应的前一个字（为以后回溯做准备）
- 回溯处理后的栅栏图，输出结果

### 实验效果
句准确率：$0.4011$  
字准确率：$0.3811$  
平均每句话时间：$0.0073s$  
测试样例总时间：$3.6718s$

### 样例分析
- 正确的样例：
  - 有些令人揪心
  - 二次元文化开始发展
  - 首都国际机场
  - 人工智能遥遥领先
  - 中国共产党员的初心和使命是为中国人民谋幸福为中华民族谋复兴
  - 以习近平同志为总书记的党中央
  - 走中国特色社会主义道路  
  对于正确的样例，不难看出有以下两个特征之一
  - 是新闻中广泛出现的词汇
  - 句子较短小且由常见的二字字串组成
- 错误的样例
  - 知道博士毕业迷茫的时候（直到博士毕业迷茫的时候）
  - 创新疆克省十几内容（创新讲课形式及内容）
  - 违纪伯克是一个网罗伯克全数项目（维基百科是一个网络百科全书项目）
  - 轻人介会了（情人节快乐）
  - 强烈谴责的劣社会分子（强烈谴责低劣社会分子）  
  不难看出，导致错误的主要原因是：
  - 常见度较低的二字词难以出现：例如上述第一句中的“直到”在语料库中的出现频率要低于“知道”，因此测试结果中相对应的也更容易出现“知道”
  - 输入的断句问题：例如上述第二句中，本应是“chuang xin/jiang ke/...”，但是在处理的过程中，“xin jiang”被认为是“新疆”，导致了错误
  - 多音字的出现很容易导致错误：因为python标准库很难区分字符在具体语境中的读音

### 参数对比 
根据测试，平滑化参数$\lambda$对结果有着很大影响，由于其代表的是单字概率，因此当$\lambda$较大($>0.01$)时，$\text{Viterbi}$算法会越来越接近甚至退化为普通的贪心算法，因此最终的准确率会非常低。当$\lambda$较小的时候，其准确率会有显著上升，最终在$1e-5$的时候到达平台期，此后再继续降低没有显著效果，反而有可能下降，我认为这可能是$\text{python}$的计算精度导致的

### 时间空间复杂度分析
时间复杂度$O(nm^{2})$，其中$n$为句子长度，$m$为其中不同拼音对应字序列长度的最大值  
空间复杂度主要来源于存储拼音-字串对照表，在二元语法下，我将其处理成了$\text{json}$格式，最终占用空间为$51.5\text{Mb}$，在模型中，需要将其读入并转化为字典  
实际运算中，算法所需的运算次数为
$$\sum\limits_{i=1}^{n}m_{i-1}m_{i}$$
其中$m_i$为第$i$个拼音对应的字的序列长度，定义$m_0 = 1$

## 基于字的三元模型输入法

### 基本思路
与二元模型相同

### 公式推导
与二元基本相同  
由于我们考虑的是三元模型，因此每个字的出现只与其前方两个字有关，有：
$$P(S) = \prod\limits_{i=2}^{n}P(w_i | w_{i-2}w_{i-1})$$
平滑化方式如下
$$(1-\eta)P(w_i | w_{i-2}w_{i-1}) + \eta((1-\lambda)P(w_{i} | w_{i-1}) + \lambda P(w_i))\rightarrow P(w_{i} | w_{i-1})$$
这样每一个字串都被赋予了一个最低值概率，其中的$\lambda, \eta$为可调参数  

### 实现过程
与二元基本相同  
处理首层时只考虑单字的概率，处理次层时考虑前两个字的概率，后续处理时均考虑三字概率  
由于完全遍历前两层所需要的时间复杂度过高，因此采用如下方法：只遍历前一层，并且将前方第二个字选为让被遍历到的前一层的字出现的概率最大的那个字，即考虑$w_{i-2}w_{i-1}w_{i}$时，只遍历第 $i-1$层，$w_{i-2}$由$w_{i-1}$唯一确定

### 实验效果
句准确率：$0.5309$  
字准确率：$0.8810$  
平均每句话时间：$0.0089s$  
测试样例总时间：$4.4637s$

### 样例分析
- 正确的样例：
  - 人工智能技术发展迅猛
  - 每隔四年一次的冬奥会在今年召开了
  - 机动车驾驶员培训手册
  - 中国共产党员的初心和使命是为中国人民谋幸福为中华民族谋复兴
  - 以习近平同志为总书记的党中央
  - 走中国特色社会主义道路
- 错误的样例
  - 知道博士毕业迷茫的时候（直到博士毕业迷茫的时候）
  - 创新疆克省实际内容（创新讲课形式及内容）
  - 大口罩防止交叉感染（戴口罩防止交叉感染）
  - 青藏大甩卖（清仓大甩卖）
  - 化化运动员与乘接线（花滑运动员羽生结弦）  
  不难看出，导致错误的主要原因仍然是输入的断句与多音字问题

### 参数对比 
根据测试，平滑化参数$\lambda, \eta$对结果有着很大影响，在测试结果中，当$\lambda = 0.0001, \eta = 0.1$时达到最优，

### 时间空间复杂度分析
时间复杂度$O(nm^{2})$，其中$n$为句子长度，$m$为其中不同拼音对应字序列长度的最大值  
空间复杂度主要来源于存储字串频率表，在三元语法下，二字频率表的大小为$16\text{Mb}$，三字频率表的大小为$63\text{Mb}$  
实际运算中，算法所需的运算次数为
$$\sum\limits_{i=1}^{n}m_{i-1}m_{i}$$
其中$m_i$为第$i$个拼音对应的字的序列长度，定义$m_0 = 1$

## 感受
通过这次作业，我深入理解了$\text{Viterbi}$算法，对于课上讲解的搜索问题也有了更加深入的理解，同时在尝试过程中，我认识到了数据集的不同对于结果的极大影响，在一开始我没有处理语料库中的多音字，也即将同一个字的读音只取其中一个（通常是字典序较大的），这也导致了最终训练结果很差，例如“qing hua da xue”会输出“清华达学”，相反“qing hua dai xue”会输出“清华大学”，在经过处理之后，这种问题有了很好的改善，但是处理方法比较简单，因此在很多情况下面对多音字的时候仍然会出错。  
同时，我也认识到了参数对于模型的影响能力，本次实验二元模型包含平滑化参数$\lambda$，这个参数可以在相当程度上影响实验准确性，在算法不变的情况下，参数可以让实验结果在$0\sim 0.4$内有很大波动，三元的情况下，$\lambda, \eta$ 可以让结果在$0.33\sim 0.53$之间波动  
在实现了二元模型之后，我也重新实现了三元模型，更优的模型拥有更高的准确率，但是也需要消耗更多的时间，这也让我认识到效率和准确率二者是很难兼得的