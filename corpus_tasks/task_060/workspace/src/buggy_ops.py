def diferenca_max_min(nums):
    if not nums:
        return 0
    return max(nums) + min(nums)
