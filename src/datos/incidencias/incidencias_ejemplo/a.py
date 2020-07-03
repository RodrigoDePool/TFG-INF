
import json
cols = {
    'numberCnx': 'Conexiones',
    'rttPerCnx': 'RTT',
    'rtx': 'Retransmisiones',
    'bpsPhyRcv': 'Bps recibidos'
}

def format(inc):
    ret = ''
    desde = inc['desde'].split()
    ret += '\\specialcell{'+ desde[0] + \
        '\\\\' + desde[1] + '}'

    
    desde = inc['hasta'].split()
    ret += '& \\specialcell{'+ desde[0] + \
        '\\\\' + desde[1] + '}'
    
    ret += '& '+cols[inc['columna']]
    try:
        ret += '& {:.4}\\%'.format(inc['proporcion']*100)
    except:
        ret += '& {}\\%'.format(inc['proporcion']*100)
    
    ret += '& '+str(inc['intensidad'])
    
    return str(ret)

file = 'up_rtx_v.json'


with open(file, 'r') as f:
    incs = json.load(f)

a='\\hline\n'
for inc in incs:
   a+=format(inc) + '\\\\\n\\hline\n'
print(a) 