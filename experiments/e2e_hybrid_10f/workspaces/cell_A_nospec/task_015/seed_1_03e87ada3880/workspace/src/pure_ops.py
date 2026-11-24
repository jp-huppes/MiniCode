def safe_div(numerator, denominator):
    if denominator != 0:
        return numerator / denominator
    else:
        return None

    def safe_div(numerator, denominator, default=None):
    if denominator != 0:
        return numerator / denominator
    else:
        return default
        if denominator != 0:
            return numerator / denominator
        else:
            return default