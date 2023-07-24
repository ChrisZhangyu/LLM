# import re
# txt = ""
# with open("新建文本文档.txt") as f:
#     for line in f.readlines():
#         line = re.search(">.*<", line)
#
#         txt = txt + line[0].strip("<>")
#
# print(txt)
#

class T:
    def __init__(self):
        self.test = 0

temp = T()

def test_arg(t):
    t.test = 1

test_arg(temp)
print(temp.test)
