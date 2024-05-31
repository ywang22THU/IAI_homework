## 文件结构
```shell
.
├── README.md
├── data
│   ├── std_input.txt
│   └── std_output.txt
└── src
    ├── run.sh
    ├── pinyin.py
    └── wash_data.py
```
- ```run.sh```为调参测试脚本
- ```pinyin.py```是模型代码
- ```wash_data.py```是数据清洗代码
## 运行方式
- 清洗数据：
  
  ```shell
  python wash_data.py <raw-data-floder>
  ```
  注：```raw-data-floder```需要是文件夹名，数据清洗器将会遍历处理其**所有直接子文件**，由于$os$包的限制，文件夹名中不能包括```../```一类的操作，例如：
  
    ```shell
    python wash_data.py ../raw-data/
    ```
    会报错，应当写成

    ```shell
    python wash_data.py /raw-data
    ```
    并且**只允许输入相对路径**，请将所有原始数据文件放在一个与$\text{src}$同级的目录下并在```src```目录下使用上方命令清洗数据，如下：

    ```shell
    .
    ├── README.md
    ├── data
    │   └── ...
    ├── raw-data
    │   └── ...
    └── src
        └── ...

    ./src
    $ python wash_data.py /raw-data
    ```
  同时，原始数据中应当至少包含音-字对照表（即下发的./拼音汉字表/拼音汉字表.txt）与语料库，格式如下：
  ```shell
  raw-data
  ├── pinyin2word.txt 名称不可变
  └── <Article Set>
  ```
  清洗完成之后，会在$\text{src}$下生成四个```json```文件，分别是：
  ```shell
  pinyin2word.json          音-字对照表
  char2counts.json          拼音-单字频率对照表
  words2counts.json         拼音-二字词频率对照表
  words3counts.json         拼音-三字词频率对照表
  ```
- 运行模型
  - 文件输入模式
  
    ```shell
    python pinyin.py 0 <input_file> <output_file> <std_output_flie>
    ```
    最后一个参数标准输出用于计算准确度，可以忽略
  - 终端输入模式
  
    ```shell
    python pinyin.py 1
    ```
    $0$为终端输入模式标识符，可以忽略，按$\text{Ctrl-C}$退出