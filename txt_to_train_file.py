#encoding=utf-8
import re
corpus_path="raw.txt"#此处请修改为原语料文件的路径
#####################################################
#语料文件格式要求：一个文章就是一行，文章与文章之间有两个"\n"，命名实体两侧分别标记"##"

with open(corpus_path,"r") as f:
    file=f.read()    
split_file=file.strip().split("\n\n")

with open(r"to_train.txt","a+") as fo:
    for article in split_file:
        is_last_article=True if split_file.index(article)==len(split_file)-1 else False#是否是最后一片文章，是的话最后就不加"\n"
        a=article
        position={}
        str_write=""
        for i in re.finditer("##.*?##",a):        
            for j in re.finditer(str(i.group()),a):
                lengh_label=len(str(i.group()).replace("#",""))
                if lengh_label>=2:#创建字典，键名为实体在字符串中索引的位置，值为如下
                    position[j.start()]="{}{}{}".format("B_PRODUCT ","I_PRODUCT "*(lengh_label-2),"E_PRODUCT ")
                else:
                    position[j.start()]="B_PRODUCT " 
            a=re.sub(str(i.group()),"",a)#把正则匹配出来的都换成空字符
        for idx in range(len(a)):#新建一个字符串，把O一个一个加进去，到实体的位置就加对应的标签
            if idx in position.keys():
                str_write+=position[idx]
            if idx !=len(a)-1:
                str_write+="O"+" " 
            else:
                str_write+="O"       
#输出训练文件说明：每个训练样本就是一行，间隔一个"\n"，直接readline()就行
        if is_last_article:
            fo.write(str_write)
        else:
            fo.write(str_write+"\n")        
    fo.close()