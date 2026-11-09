
PROMPT_TEMPLATE = "Q: What month is {} months after{}?\nA:"
MONTHS = [
    'January', 
    'February', 
    'March', 
    'April', 
    'May', 
    'June', 
    'July', 
    'August', 
    'September', 
    'October', 
    'November', 
    'December'
]

OFFSETS = [
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven'
]

def get_months_dataset():
    texts = []
    ans = []

    for j, month in enumerate(MONTHS):
        for i, offset in enumerate(OFFSETS):
            texts.append(PROMPT_TEMPLATE.format(offset, month))
            ans.append(MONTHS[(i+j+1)%12])

    return texts, ans