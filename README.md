# Siri Sign

**A língua de sinais como interface do Siri.** Protótipo acadêmico (Insper) que imagina o
Siri entendendo a Língua Brasileira de Sinais (Libras): você sinaliza para a câmera e o
"Siri" responde com a palavra reconhecida.

### ▶︎ Abrir o protótipo: **https://juliadn.github.io/libras/**

Funciona direto no navegador do computador ou do celular (Safari ou Chrome), sem instalar
nada. Toque em **Ativar câmera** e permita o acesso. Nenhuma imagem sai do aparelho: todo
o reconhecimento roda no próprio navegador.

## Sinais para testar

| Sinal | Tipo |
|---|---|
| **Arroz** | com movimento |
| **Feijão** | com movimento |
| **Nome** | com movimento |
| **Julia** | soletrado J · U · L · I · A (use o modo *Soletrar*) |
| Sinal, Casa | também reconhecidos |

Dicas: fique a cerca de 1 m da câmera com rosto e ombros visíveis, levante as mãos acima
da linha pontilhada, faça o sinal e pare ou abaixe as mãos no final.

## Como funciona

1. **MediaPipe Holistic** (Google) encontra, a cada quadro, 21 pontos de cada mão, os
   ombros e cotovelos e 40 pontos do rosto (sobrancelhas, olhos, boca).
2. Os pontos são **normalizados** pelo corpo: o formato de cada mão em relação ao próprio
   pulso, a posição das mãos em larguras de ombro e as expressões em relação ao nariz.
   Assim não importa onde a pessoa está no quadro nem a distância da câmera.
3. Um **segmentador** decide quando um sinal começa (mão acima da linha) e termina (mãos
   paradas por 0,4 s ou abaixadas).
4. Um classificador **k-NN** compara a tentativa com os exemplos gravados de cada sinal
   (letras paradas pela forma média; sinais com movimento reamostrados em 32 quadros) e
   responde só quando a confiança passa de 50%. Caso contrário: "Não entendi".

Não há rede neural treinada: ensinar um sinal novo é só gravar alguns exemplos.

## Estrutura do repositório

```
docs/      o site publicado no GitHub Pages (HTML + CSS + JS, sem build)
  js/engine.js        reconhecimento (porte linha a linha do Python)
  js/app.js, orb.js   câmera, interface e a "orb" do Siri
  data/               exemplos gravados, já convertidos em descritores
  models/             modelo MediaPipe Holistic (13 MB)
python/    a versão original local em Python (FastAPI), usada para gravar os sinais
  tools/export_web.py   gera docs/data a partir dos exemplos gravados
  tools/web_parity.*    teste: o JS dá os mesmos resultados que o Python
```

A versão do site foi verificada contra a versão Python: com os mesmos pontos de entrada,
os descritores coincidem (diferença < 10⁻¹⁶) e as previsões são idênticas.

---

<sub>Protótipo acadêmico · Insper · Conceito não afiliado à Apple. Siri é marca da Apple Inc.</sub>
