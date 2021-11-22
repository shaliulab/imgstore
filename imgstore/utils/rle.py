# Python code for run length encoding
def encode_message(message):
    encoded_string = ""
    i = 0
    while (i <= len(message)-1):
        count = 1
        ch = message[i]
        j = i
        while (j < len(message)-1): 
            '''if the character at the current index is the same as the character at the next index. If the characters are the same, the count is incremented to 1'''    
            if (message[j] == message[j + 1]): 
                count = count + 1
                j = j + 1
            else: 
                break
        '''the count and the character is concatenated to the encoded string'''
        encoded_string = encoded_string + str(count) + ch
        i = j + 1
    return encoded_string

def get_rle(data):
    encoded = encode_message(data)
    values = []
    lengths = []
    accum = []
    count = ""
    for i in range(len(encoded)):
        c = encoded[i]
        try:
            int(c)
            count += c
        except ValueError:
            values.append(c)
            lengths.append(int(count))
            accum.append(sum(lengths))
            count = ""
    
    return values, accum


if __name__ == "__main__":
    encoded = encode_message(["T", "T", "F","T"])
