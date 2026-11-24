
def parse_int_list(csv_str):
    return [int(x) for x in csv_str.split(',') if x.strip()]


def parse_int_list(csv_str):
    # Remove leading and trailing whitespace
    cleaned_csv = csv_str.strip()
    
    # Split the CSV by commas
    items = cleaned_csv.split(',')
    
    # Convert each item to an integer
    try:
        return [int(item) for item in items]
    except ValueError:
        return []
def parse_int_list(csv_str):
    # Remove leading and trailing whitespace
    cleaned_csv = csv_str.strip()
    
    if not cleaned_csv:
        return []
    
    # Split the CSV by commas
    items = cleaned_csv.split(',')
    
    # Convert each item to an integer
    try:
        return [int(item) for item in items]
    except ValueError:
        raise ValueError('Invalid input: non-integer values found')

def parse_int_list(csv_str):
    # Remove leading and trailing whitespace
    cleaned_csv = csv_str.strip()
    
    # Split the CSV by commas
    items = cleaned_csv.split(',')
    
    # Convert each item to an integer
    return [int(item) for item in items]