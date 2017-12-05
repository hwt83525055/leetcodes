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
