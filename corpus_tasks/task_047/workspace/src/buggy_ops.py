def maior_elemento(nums):
    maior = 0
    for n in nums:
        if n > maior:
            maior = n
    return maior
