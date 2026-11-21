def media_segura(nums):
    if not nums:
        return 0.0
    return sum(nums) // len(nums)
