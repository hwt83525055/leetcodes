from functools import reduce
class Solution:
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