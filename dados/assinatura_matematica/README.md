# Dados congelados da assinatura matematica

Versao de desenvolvimento: 1.18.0-dev.1

Estes CSVs congelam apenas as observacoes contextuais necessarias para a
modelagem matematica. Nenhum replay novo foi usado para produzi-los nesta
etapa.

## contextual_batch3.csv

Origem: pacote do terceiro lote, arquivo `signature_batch3_candidates.csv`.

- 20 candidatos;
- features calculadas contra U56;
- efeito financeiro usado como alvo medido contra U59;
- positivos: MG, REXR e CALM;
- SHA-256 do CSV canonico:
  `f62257f0a9b86d91ee1d7c3143f68edf12a0154edc45db1d345102bcd219445a`.

A diferenca entre referencia das features (U56) e referencia financeira (U59)
e explicitamente preservada nas colunas `feature_reference` e
`financial_reference`. A modelagem usa ranks dentro da coorte para reduzir,
mas nao eliminar, essa incompatibilidade de escala/contexto.

## contextual_smart20.csv

Origem: lista congelada do search 1.17.0 e diagnostico financeiro individual
da mesma lista.

- 20 candidatos;
- features calculadas contra U59;
- efeito financeiro medido contra U59;
- 8 positivos, 1 neutro e 11 negativos;
- SHA-256 do CSV canonico:
  `60be7360ffcf56abbd434fe7a14364e6afdfc5b50d1887793f510024ab3f6b28`.

## Controles auxiliares

O runner `modelar_assinatura_matematica_spyder.py` usa, quando presentes:

- `output/busca_ativos/intelligent_known_training.csv` como controle negativo
  de features estaticas entre batch2 e batch3;
- `output/analise_standalone_ativos_58m/standalone_trade_details.csv` como
  camada temporal explicativa dos trades dos oito vencedores.

Esses dois arquivos auxiliares nao entram no score contextual principal. Se
estiverem ausentes, o score principal ainda e executado e a ausencia e
registrada no log.

## Regra cientifica

Os dados deste diretorio sao DEVELOPMENT DATA. A formula produzida por 1.18
nao pode ser chamada de validacao externa. Depois de congelada, ela devera ser
testada uma unica vez em candidatos intocados.
