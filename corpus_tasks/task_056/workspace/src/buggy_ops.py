def contar_maiores(nums, corte):
    return sum(1 for n in nums if n >= corte)
