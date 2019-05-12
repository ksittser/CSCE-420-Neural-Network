# Kevin Sittser
# 525003900
# CSCE 420
# Due: April 25, 2019
# bdfparse.py


'''Parse the BDF files into binary number arrays for each letter'''

def hexToBin(hexNum):
    '''Return the eight-digit binary representation of input (two-digit) hexadecimal number'''
    return bin(int(hexNum,16))[2:].zfill(8) + '0'

def letterMap():
    '''Return map containing a dictionary which has each uppercase letter mapped to an array of rows of binary-number strings'''
    f = open('bdf-files/ie9x14u.bdf')
    lines = f.readlines()
    maps = {}
    lineNum = 0
    while not 'Z' in maps and lineNum < len(lines):
        if len(lines[lineNum]) >= 11 and lines[lineNum][0:9] == 'STARTCHAR' and len(lines[lineNum][10:]) == 2 and lines[lineNum][10].isupper():
            letter = lines[lineNum][10]
            maps[letter] = []
            while lines[lineNum][0:6] != 'BITMAP':
                lineNum += 1
            lineNum += 1
            while lines[lineNum][0:7] != 'ENDCHAR':
                maps[letter].append(hexToBin(lines[lineNum][:-1]))
                lineNum += 1
            while len(maps[letter]) < 14:
                maps[letter].append(hexToBin('0'))
        else:
            lineNum += 1
    return maps
    
if __name__ == '__main__':
    print(letterMap())