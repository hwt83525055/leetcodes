class Solution:
    def uniquePaths2(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m == 1 or n == 1:
            return 1
        if m == 2 or n == 2:
            return max(m, n)
        ans  = [[1 for d in range(1, n+1)]]
        for k in range(1, m):
            ans.append([1])
            for r in range(1, n):
                ans[k].append(ans[k-1][r]+ans[k][r-1])
        return ans
