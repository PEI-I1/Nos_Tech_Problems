keywords = {
    'common': {
        'equipamento': [r'box', r'router', r'televis(ã|a)o', r'telefone', r'tv', r'internet'],
        'wireless': [r'wifi', r'(inter)?net', r'rede', r'liga(ç|c)(ã|a)o'],
        'ruído': [r'barulhento', r'barulhenta'],
        'degradação': [r'piorar', r'pior[\n\s\b]', r'a piorar'],
        'ecrã': [r'televis(ã|a)o', r'tv'],
        'negro': [r'pret(o|a)', r'escur(o|a)'],
        'conteúdo': [r'cana(l|is)', r'filme', r'videoclube'],
        'app': [r'(\b|^)programa(\b|$)', r'aplica(ç|c)(ão|ao|ões|oes)'],
        'apps': [r'(\b|^)programa(\b|$)', r'aplica(ç|c)(ão|ao|ões|oes)'],
        'acesso': [r'(\b|^)aceder(\b|$)'],
        'no acesso': [r'(\b|^)ao aceder'],
        'indisponível': [r'(\b|^)n(ã|o)o consigo(\b|$)'],
        'login': [r'autentica(r|ção|cao|çao|cão)', r'(\b|^)iniciar sess(ã|a)o(\b|$)'],
        'chamada': [r'(\b|^)ligar(\b|$)'],
        'quebras': [r'falha(s|r)'],
        'ftth': [r'fibra'],
        'tv': [r'tv', r'televis(ã|a)o', r'image(m|ns)']
    },
    'tip_1':{
        'internet': [r'wifi', r'net', r'rede', r'wireless', r'ethernet'],
        'gravação': [r'gravar'],
        'guia': [r'menu', r'op(ç|c)(ão|ao|ões|ões)'],
        'restart': [r'reiniciar', r'ligar? e desligar?']
    },
    'tip_2':{
        'imagem/áudio': [r'(\b|^)som(\b|$)', r'(á|a)udio', r'imagem', r'cana(l|is)'],
        'visualizar': [r'(\b|^)ver(\b|$)', r'(\b|^)vejo(\b|$)'],
        'áudio': [r'(\b|^)som(\b|$)'],
        'emissão': [r'tv', r'televis(ã|a)o', r'cana(l|is)'],
        'booting': [r'(\b|^)a ligar(\b|$)'],
        'inst./download': [r'instala(r|ção|çao|cão|cao)', r'download', r'descarregar', r'transferir'],
        'incorreta': [r'mal', r'má(s)?'],
        'intermitência': [r'interrup(ç|c)(ão|ao|ões|oes)', r'(\b|^)com paragens(\b|$)', r'(\b|^)a parar(\b|$)'],
        'internacional': [r'estrangeiro', r'(\b|^)l(á|a) para fora(\b|$)', r'(\b|^)fora(\b|$)']
    },
    'tip_3':{
        'software': [r'programas?', r'sistema'],
        'avaria': [r'estragado', r'(\b|^)n(ã|a)o (est(á|a) a )?funcionar?(\b|$)'],
        'esporádico': [r'de vez em quando', r'(à|a)s vezes'],
        'configuração': [r'menu', r'defini(ç|c)(ões|oes|ão|ao)', r'altera(ç|c)(ã|a)o'],
        'reboots': [r'reiniciar', r'ligar? e desligar?'],
        'specs': [r'contrat(ad)?o'],
        'visualiz.': [r'(\b|^)ver(\b|$)', r'(\b|^)vejo(\b|$)'],
        'incorreto': [r'maus?', r'errados?'],
        'hw/ sw': [r'h(ard)?w(are)?', r's(oft)?w(are)?', r'equipamentos?']
    }
}
