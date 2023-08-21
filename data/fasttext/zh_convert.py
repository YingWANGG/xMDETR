import chinese_converter

data = []
with open('en-zh_raw.txt') as f:
    for line in f:
        line = line.split()
        en,zh = line[0],line[1]
        zh = chinese_converter.to_simplified(zh)     
        data.append(' '.join([en,zh]))
with open('en-zh.txt','w') as f:
    for line in data:
        f.write(line)
        f.write('\n')
