import re
from typing import List

re_swu = {
    'symbol': r'[\U00040001-\U0004FFFF]',
    'coord': r'[\U0001D80C-\U0001DFFF]{2}',
    'sort': r'\U0001D800',
    'box': r'\U0001D801-\U0001D804'
}
re_swu['prefix'] = rf"(?:{re_swu['sort']}(?:{re_swu['symbol']})+)"
re_swu['spatial'] = rf"{re_swu['symbol']}{re_swu['coord']}"
re_swu['signbox'] = rf"{re_swu['box']}{re_swu['coord']}(?:{re_swu['spatial']})*"
re_swu['sign'] = rf"{re_swu['prefix']}?{re_swu['signbox']}"
re_swu['sortable'] = rf"{re_swu['prefix']}{re_swu['signbox']}"


def fsw2data(fswText: str) -> str:
    match = fswText[re.search('[MBLR]', fswText).start():]  # remove signs prefix
    match = match[re.search('[S]', match).start():]  # remove Box positional prefix
    match = match.split('S')[1:]  # split the string into a list of sign
    match = "x".join(match).split('x')  # split the string into a list of sign
    # if the len maych[i] bigget then 5 split it to 2 part without for loop
    data = []
    for i in range(len(match)):
        if len(match[i]) > 5:
            data.append(match[i][:5])
            data.append(match[i][5:])
        else:
            data.append(match[i])
    return " ".join(data)


def data2fsw(dataText: str) -> str:
    signs = ['A']
    dataText = dataText.split(' ')
    for i in dataText:
        if len(i) >= 5:
            signs.append(i)
            dataText.remove(i)
    fsw = "S".join(signs) + "M514x517"
    for index, val in enumerate(signs[1:]):
        fsw += "S" + val + dataText[2 * index] + "x" + dataText[2 * index + 1]
    return fsw

# from signbank-plus
def swu2key(swuSym: str) -> str:
    symcode = ord(swuSym) - 0x40001
    base = symcode // 96
    fill = (symcode - (base * 96)) // 16
    rotation = symcode - (base * 96) - (fill * 16)
    return f'S{hex(base + 0x100)[2:]}{hex(fill)[2:]}{hex(rotation)[2:]}'

# from signbank-plus
def swu2coord(swuCoord: str) -> List[int]:
    return [swu2num(swuCoord[0]), swu2num(swuCoord[1])]

# from signbank-plus
def swu2num(swuNum: str) -> int:
    return ord(swuNum) - 0x1D80C + 250

# from signbank-plus
def swu2fsw(swuText: str) -> str:
    if not swuText:
        return ''

    # Initial replacements
    fsw = swuText.replace("𝠀", "A").replace("𝠁", "B").replace("𝠂", "L").replace("𝠃", "M").replace("𝠄", "R")

    # SWU symbols to FSW keys
    syms = re.findall(re_swu['symbol'], fsw)
    if syms:
        for sym in syms:
            fsw = fsw.replace(sym, swu2key(sym))

    # SWU coordinates to FSW coordinates
    coords = re.findall(re_swu['coord'], fsw)
    if coords:
        for coord in coords:
            fsw = fsw.replace(coord, 'x'.join(map(str, swu2coord(coord))))

    return fsw

# from signbank-plus
def swu2data(swuText: str) -> str:
    return fsw2data(swu2fsw(swuText))


if __name__ == "__main__":
    print(data2fsw("18711 490 483 20500 486 506"))
    print(fsw2data("AS18711S20500M514x517S18711490x483S20500486x506").split(' '))
