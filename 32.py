class Solution(object):
    def longestValidParentheses(self, s):
        dp = [0,]
        for m in range(1, len(s)):
            if s[m] == '(':
                dp.append(0)
            else:
                if s[m - 1] == '(':
                    dp.append(2 + dp[m-2] if m>2 else 2)
                else:
                    if (m-dp[m-1]-1)>=0:
                        if s[m-dp[m-1]-1] == '(':
                            if m-dp[m-1]-1>0:
                                dp.append(2 + dp[m-1] + dp[m - dp[m-1] - 2])
                            else:
                                dp.append(2 + dp[m-1])
                        else:
                            dp.append(0)
                    else:
                        dp.append(0)
        return max(dp)
s = Solution()
print(s.longestValidParentheses('(((()))))()()()()()'))



# class Solution(object):
#
#     @classmethod
#     def largestRectangleArea(self, height):
#         height.append(0)
#         stack = [0]
#         r = 0
#         for i in range(1, len(height)):
#             while stack and height[i] < height[stack[-1]]:
#                 h = height[stack.pop()]
#                 w = i if not stack else i - stack[-1] - 1
#                 r = max(r, w * h)
#             stack.append(i)
#         return r

# print(Solution.largestRectangleArea([6, 3, 4, 5, 6, 5, 3, 1]))

