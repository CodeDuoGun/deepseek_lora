"""给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

"""
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        if len(s) > len(t):
            return False
        low = 0
        for fast in range(len(t)):
            if low >= len(s):
                break
            if s[low] == t[fast]:
                low +=1
        if low < len(s):
            return False
        return True 

ss = Solution()
print(ss.isSubsequence("aed", "abcde"))

        