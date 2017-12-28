# class Solution:
#     def sort(self, dic):
#         dic_new = {}
#         for k in sorted(dic.keys()):
#             dic_new[k] = dic[k]
#         return dic_new
#     def minWindow(self, s, t):
#         """
#         :type s: str
#         :type t: str
#         :rtype: str
#         """
#         dic = {}
#         for char in s:
#             if char in t:
#                 dic[char] = []
#         dic_need = self.sort({char:t.count(char) for char in t})
#         for char in range(len(s)):
#             if s[char] in t:
#                 dic[s[char]].append(char)
#         dic = self.sort(dic)
#         for key in dic_need.keys():
#             if key not in dic.keys():
#                 return ''
#             if dic[key].__len__()<dic_need[key]:
#                 return ''
#         values = list(dic.values())
#         start = [[s for s in range(dic_need[k])] for k in dic_need]
#         ans = s
#         now_rank = []
#         for elem in range(start.__len__()):
#             now_rank.append(values[elem][start[elem][0]:start[elem][-1] + 1] if start[elem].__len__() > 1 else [
#                 values[elem][start[elem][0]]])
#         while True:
#             ans = min(ans, s[min(now_rank,key=lambda x:min(x))[0]:max(now_rank,key=lambda x:max(x))[-1]+1], key=lambda x:len(x))
#             min_index = now_rank.index(min(now_rank, key=lambda x:x[0]))
#             for d in range(len(start[min_index])):
#                 start[min_index][d]+=1
#             if start[min_index][-1]>values[min_index].__len__()-1:
#                 break
#             now_rank = []
#             for elem in range(start.__len__()):
#                 now_rank.append(values[elem][start[elem][0]:start[elem][-1] + 1] if start[elem].__len__() > 1 else [
#                     values[elem][start[elem][0]]])
#         return ans

from functools import reduce
import collections
class Solution:
    # def minWindow(self, s, t):
    #     need, missing = collections.Counter(t), len(t)
    #     i = I = J = 0
    #     for j, c in enumerate(s, 1):
    #         missing -= need[c] > 0
    #         need[c] -= 1
    #         if not missing:
    #             while i < j and need[s[i]] < 0:
    #                 need[s[i]] += 1
    #                 i += 1
    #             if not J or j - i <= J - I:
    #                 I, J = i, j
    #     return s[I:J]
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        for elem in t:
            if s.count(elem)<t.count(elem):
                return ""
        ans = s
        template = {elem:{'needs':t.count(elem), 'arr': []} for elem in t}
        status = 0
        for elem in range(len(s)):
            flag = False
            if s[elem] in t:
                if template[s[elem]]['arr'].__len__() == template[s[elem]]['needs']:
                    temp = template[s[elem]]['arr'].pop(0)
                    template[s[elem]]['arr'].append(elem)
                else:
                    template[s[elem]]['arr'].append(elem)
                    temp = elem
                if not status:
                    for key in template.keys():
                        if len(template[key]['arr']) != template[key]['needs']:
                            flag = True
                            break
                    else:
                        status = 1
                    if flag:
                        continue
                if 'ping' in locals():
                    ping[ping.index(temp)] = elem
                    ans = min(ans, s[min(ping):max(ping) + 1], key=lambda k: len(k))
                else:
                    if reduce(lambda a, b: a and b,
                              list(map(lambda k: len(k['arr']) == k['needs'], template.values()))):
                        ping = []
                        for tem in template.values():
                            ping += tem['arr']
                        ans = s[min(ping):max(ping) + 1]
        return ans
s = Solution()
print(s.minWindow('ABCDEFASDVVVXXAADC', 'AADCVVVXX'))