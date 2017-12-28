# class Solution:
#     def permuteUnique(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         res = []
#         nums = sorted(nums)
#         has_u = [False for i in nums]
#
#     def dfs(self, res, has_u, k, nums):
#         for i in range(len(nums)):
#             if has_u[i] == True:
#                 return
#             if i>0 and nums[i] == nums[i-1] and has_u[i-1] == True:
#                 return
#
# s = Solution()
# print(s.permuteUnique([3,2,3,2,1]))
'''
sort_quick
'''

def quick_sort(nums, start, end):
    '''
    :param nums: List[int]
    :param start: int
    :param end: int
    :return: List[Int]
    '''