keywords = {
    'common': {
        'equipamento': [r'box', r'router', r'televisão', r'telefone', r'tv', r'internet'],
        'wireless': [r'wifi', r'internet', r'rede', r' net', r'ligação'],
        'ruído': [r'barulhento', r'barulhenta'],
        'degradação': [r'piorar', r'pior[\n\s\b]', r'a piorar'],
        'ecrã': [r'televisão', r'tv'],
        'negro': [r'pret(o|a)', r'escur(o|a)'],
        'conteúdo': [r'cana(l|is)', r'filme', r'videoclube'],
        'app': [r'(\b|^)programa(\b|$)', r'aplicaç(ão|ões)'],
        'apps': [r'(\b|^)programa(\b|$)', r'aplicaç(ão|ões)'],
        'acesso': [r'(\b|^)aceder(\b|$)'],
        'no acesso': [r'(\b|^)ao aceder'],
        'indisponível': [r'(\b|^)não consigo(\b|$)'],
        'login': [r'autentica(r|ção)', r'(\b|^)iniciar sessão(\b|$)'],
        'chamada': [r'(\b|^)ligar(\b|$)'],
        'quebras': [r'falha(s|r)'],
    },
    'tip_1':{
        'internet': [r'wifi', r'net', r'rede', r'wireless', r'ethernet'],
        'gravação': [r'gravar'],
        'guia': [r'menu', r'opç(ão|ões)'],
        'restart': [r'reiniciar', r'ligar? e desligar?']
    },
    'tip_2':{
        'imagem/áudio': [r'(\b|^)som(\b|$)', r'áudio', r'imagem', r'cana(l|is)'],
        'visualizar': [r'(\b|^)ver(\b|$)', r'(\b|^)vejo(\b|$)'],
        'áudio': [r'(\b|^)som(\b|$)'],
        'emissão': [r'tv', r'televisão', r'cana(l|is)'],
        'booting': [r'(\b|^)a ligar(\b|$)'],
        'inst./download': [r'instala(r|ção)', r'download', r'descarregar', r'trasnferir'],
        'incorreta': [r'mal', r'má(s)?'],
        'intermitência': [r'interrupç(ão|ões)', r'(\b|^)com paragens(\b|$)', r'(\b|^)a parar(\b|$)'],
        'internacional': [r'estrangeiro', r'(\b|^)lá para fora(\b|$)', r'(\b|^)fora(\b|$)']
    },
    'tip_3':{
        'software': [r'programas?', r'sistema'],
        'avaria': [r'estragado', r'(\b|^)não (está a )?funcionar?(\b|$)'],
        'esporádico': [r'de vez em quando', r'às vezes'],
        'configuração': [r'menu', r'definiç(ões|ão)', r'alteração'],
        'reboots': [r'reiniciar', r'ligar? e desligar?'],
        'specs': [r'contrat(ad)?o'],
        'visualiz.': [r'(\b|^)ver(\b|$)', r'(\b|^)vejo(\b|$)'],
        'incorreto': [r'maus?', r'errados?']
    }
}
