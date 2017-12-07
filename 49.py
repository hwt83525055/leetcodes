class Solution:
    def hashstr(self, string):
        length = len(string)
        a1 = 1
        a2 = 0
        for char in string:
            a1 *= ord(char)
            a2 += ord(char)
        return a1 + a2*10000

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dic = {}
        for str in strs:
            if self.hashstr(str) in dic.keys():
                dic[self.hashstr(str)].append(str)
            else:
                dic[self.hashstr(str)] = [str]
        return list(dic.values())
    