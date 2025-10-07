- Os valores são limitados entre 0 e 100%.

- **Saída**:
- Gráficos e tabelas com variação da umidade, lâminas aplicadas e balanço hídrico.

---

## **7. Arquivo: `CompareProductivity.py`**

Realiza a comparação entre as estratégias **RBS** e **RL** utilizando simulações do AquaCrop-OSPy.

- **Funcionalidade**:
- Carrega os cronogramas de irrigação gerados pelos métodos RBS e RL.
- Executa simulações no AquaCrop para ambas as estratégias e calcula:
  - Produtividade fresca
  - Produtividade seca
  - Produtividade potencial
  - Irrigação total sazonal (mm)

- **Etapas**:
1. Insere os valores de irrigação diária na matriz `ITN` do AquaCrop.
2. Executa as simulações para cada método sob as mesmas condições climáticas.
3. Gera gráficos de produtividade versus uso de água.

- **Saída**:
- Relatório consolidado comparando:
  - Eficiência do RL.
  - Economia de água em relação ao RBS.
  - Resposta produtiva ao volume de irrigação aplicado.

---

# **Requisitos do Projeto**

Para executar os experimentos:

```bash
pip install aquacrop gym numpy pandas torch matplotlib seaborn rule-engine openpyxl
