# TODO, dictionary for each topology and common
'''keywords = {
    "common": {
        "equipamento": [],
        ...
    },
    "tip_1":{
        ...
    },
    "tip_2":{
        ...
    },
    "tip_3":{
        ...
    }
}
'''

keywords = {
    'equipamento': [r'box', r'router', r'televisão', r'telefone', r'tv', r'internet'],
    'wireless': [r'wifi', r'internet', r'rede', r' net', r'ligação'],
    'ruído': [r'barulhento', r'barulhenta'],
    'degradação': [r'piorar', r'pior[\n\s\b]', r'a piorar'],
    'áudio': [r'(\b|^)som(\b|$)'],
    'imagem/áudio': [r'(\b|^)som(\b|$)', r'áudio', r'imagem', r'cana(l|is)'],
    'ecrã': [r'televisão', r'tv'],
    'negro': [r'pret(o|a)', r'escur(o|a)'],
    'visualizar': [r'(\b|^)ver(\b|$)', r'(\b|^)vejo(\b|$)'],
    'emissão': [r'tv', r'televisão', r'cana(l|is)'],
    'conteúdo': [r'cana(l|is)', r'filme', r'videoclube'],
    'booting': [r'(\b|^)a ligar(\b|$)'],
    'inst./download': [r'instala(r|ção)', r'download', r'descarregar', r'trasnferir'],
    'app': [r'(\b|^)programa(\b|$)', r'aplicaç(ão|ões)'],
    'acesso': [r'(\b|^)aceder(\b|$)'],
    'no acesso': [r'(\b|^)ao aceder'],
    'indisponível': [r'(\b|^)não consigo(\b|$)'],
    'incorreta': [r'mal', r'má(s)?'],
    'intermitência': [r'interrupç(ão|ões)', r'(\b|^)com paragens(\b|$)', r'(\b|^)a parar(\b|$)'],
    'login': [r'autentica(r|ção)', r'(\b|^)iniciar sessão(\b|$)'],
    'internacional': [r'estrangeiro', r'(\b|^)lá para fora(\b|$)', r'(\b|^)fora(\b|$)'],
    'chamada': [r'(\b|^)ligar(\b|$)'],
    'quebras': [r'falha(s|r)']
}