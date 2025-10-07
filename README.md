# Este código faz parte do trabalho de Doutorado em Engenharia Elétrica e Informática Industrial (CPGEI - UTFPR)

Autores: **G. de Queiroz Pereira**, **D. Paulo Bertrand Renaux** e **A. E. Lazzaretti**  
Título: **COMPARATIVE ANALYSIS OF REINFORCEMENT LEARNING AND RULE-BASED SYSTEM APPROACHES FOR IRRIGATION IN HORTICULTURE**

---

# Estrutura do Projeto

## **1. Arquivo: `RBS-1.py`**

Este arquivo implementa a primeira versão do **Sistema Baseado em Regras (RBS)**, responsável por calcular o cronograma de irrigação com base no balanço hídrico e nos parâmetros da cultura.

- **Funcionalidade**:
  - Define constantes como textura do solo, tipo de cultura, eficiência de irrigação e coeficientes de cultivo (Kc) por fase.
  - Calcula as necessidades diárias de irrigação (`IRN` e `ITN`) usando o método de balanço hídrico.
  - Utiliza dados climáticos (temperatura, precipitação) para calcular a ETo pela equação simplificada de Hargreaves.

- **Principais Funções**:
  - `calcular_lamina_irrigacao()` — determina a lâmina de irrigação diária e por fase.
  - `aplicar_regras()` — aplica regras condicionais via *rule engine* para decidir se a irrigação ocorre (ex.: não irrigar se `ITN < 5`, limitar se `ITN > 40`).

- **Saída**:
  - Gera uma tabela com o cronograma diário de irrigação contendo ETo, ETc, Pe e horário de irrigação.

---

## **2. Arquivo: `RBS-2.py`**

Versão avançada do **Sistema Baseado em Regras (RBS)**, integrando dados dinâmicos de sensores e previsões meteorológicas.

- **Funcionalidade**:
  - Estende o RBS básico integrando dados de **previsão** e **sensor** para decisões em tempo real.
  - Usa a biblioteca `rule_engine` para aplicar regras lógicas (ex.: temperatura do solo e do ar).
  - Permite análises por hora, oferecendo maior resolução temporal para o controle da irrigação.

- **Integração de Dados**:
  - Lê arquivos como `dados_clima.txt` e `dados_sensor.txt`.
  - Combina medições reais de sensores (umidade, temperatura, umidade relativa) com previsões diárias.

- **Lógica de Decisão**:
  - Ajusta automaticamente o volume de irrigação.
  - Considera fatores como temperatura elevada ou solo seco para aplicar irrigações adicionais.

- **Saída**:
  - Relatório diário com volume de irrigação ajustado, condições ambientais e explicações qualitativas das decisões.

---

## **3. Arquivo: `RL-QL.py`**

Implementa o algoritmo de **Q-Learning (QL)** integrado ao modelo AquaCrop-OSPy.

- **Funcionalidade**:
  - O ambiente utiliza dados climáticos diários (`MinTemp`, `MaxTemp`, `Precipitation`, `ReferenceET`) como variáveis de estado.
  - O agente de Q-Learning aprende políticas de irrigação que maximizam a produtividade e a eficiência no uso da água.
  - As recompensas são obtidas por meio da simulação no AquaCrop, considerando o rendimento final da cultura.

- **Arquitetura**:
  - Classe `Agent`: define a Q-Table, a política *epsilon-greedy* e a atualização de Bellman.
  - Função `run_aquacrop_slice()`: executa a simulação AquaCrop em cada episódio e calcula o retorno (rendimento seco).

- **Etapas de Treinamento**:
  1. Carrega os dados climáticos de `dados_DV.xlsx`.
  2. Executa simulações do AquaCrop para cada episódio.
  3. Atualiza os valores de Q com base na produtividade e penalizações por desperdício de água.

- **Saída**:
  - Gera valores Q treinados e métricas de desempenho por episódio.

---

## **4. Arquivo: `RL-DQL.py`**

Implementa o algoritmo de **Deep Q-Learning (DQL)** com rede neural para políticas contínuas de irrigação.

- **Funcionalidade**:
  - Substitui a Q-Table discreta por uma rede neural (`DQN`) para aproximação de função.
  - Aprende políticas não lineares complexas e generalizáveis para diferentes cenários climáticos.
  - Cada estado contém informações sobre precipitação, ET₀, estágio fenológico e umidade.

- **Estrutura da Rede**:
  - Camadas totalmente conectadas com ativações ReLU (128–128 neurônios).
  - Otimização via Adam e função de perda MSE.

- **Processo de Aprendizado**:
  - Utiliza *replay memory* (`deque`) para estabilizar o treinamento.
  - A exploração diminui progressivamente com a redução de `epsilon`.

- **Integração**:
  - O ambiente interage com o AquaCrop para obter o rendimento seco (`Dry yield (tonne/ha)`), que gera a recompensa.

- **Saída**:
  - Modelo profundo (`DQN`) treinado, capaz de estimar o volume ótimo de irrigação diária.

---

## **5. Arquivo: `RL-PPO.py`**

Implementa o método **Proximal Policy Optimization (PPO)** — um algoritmo de gradiente de política para controle contínuo integrado ao AquaCrop-OSPy.

- **Funcionalidade**:
  - O agente interage com o ambiente realista (`AquaCropRealEnv`), recebendo estados, recompensas e limites de irrigação.
  - PPO aprende as redes **ator (política)** e **crítico (valor)** para estabilizar as atualizações.
  - Inclui penalizações por irrigação excessiva em dias chuvosos e bônus por economia de água.

- **Componentes do Modelo**:
  - **ActorNetwork**: gera a média e desvio padrão da quantidade de irrigação.
  - **CriticNetwork**: estima o valor do estado para o cálculo da vantagem.

- **Função de Recompensa**:
  - Recompensa maior produtividade e penaliza excesso de água e estresse hídrico.
  - Inclui penalidade por chuva (`rain_w=0.02`) e bônus quando o agente evita irrigar em condições adequadas.

- **Hiperparâmetros**:
  - Taxa de aprendizado = 2e-4, γ = 0.99, λ = 0.95, *clip* = 0.15, *entropy* = 0.05, *batch size* = 32.

- **Saída**:
  - Modelo PPO treinado, com política suave e eficiente que se adapta a diferentes condições climáticas.

---

## **6. Arquivo: `irrigation_soil.py`**

Integra o AquaCrop para calcular a irrigação com base na **umidade do solo**.

- **Funcionalidade**:
  - Monitora o armazenamento de água no solo e calcula o percentual de umidade em relação à capacidade de campo (`FC`) e ao ponto de murcha (`PWP`).
  - Define camadas de solo e parâmetros de cultura personalizados.
  - Gera gráficos da evolução da umidade e fluxos de água no solo.

- **Equação Principal**:
