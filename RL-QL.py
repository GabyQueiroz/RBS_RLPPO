# Importação de bibliotecas e módulos necessários
import os
os.environ['DEVELOPMENT'] = 'True'
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
from gym import spaces

# Carregar o arquivo Excel com os dados climáticos históricos (para o treinamento)
file_path_train = 'dados_DV.xlsx'  # Arquivo de dados climáticos históricos para o treinamento
df_climate_train = pd.read_excel(file_path_train)
df_climate_train['Date'] = pd.to_datetime(df_climate_train['Date'], format='%Y-%m-%d')

# Carregar o arquivo Excel com os dados climáticos diários (para as previsões)
file_path_test = 'dados_climaticos_diarios.xlsx'  # Arquivo de dados climáticos diários para as previsões
df_climate_test = pd.read_excel(file_path_test)
df_climate_test['Date'] = pd.to_datetime(df_climate_test['Date'], format='%Y-%m-%d')

# Definindo as datas de início e fim da simulação
sim_start = '2007/05/01'
sim_end = '2007/12/31'

# Definir o solo personalizado com base nas propriedades fornecidas
custom_soil = Soil('custom', cn=46, rew=7)

# Adicionar as camadas do solo com suas propriedades físicas e hídricas
custom_soil.add_layer(
    thickness=0.1,  # Espessura da camada em metros (0-10 cm)
    thWP=0.268,  # Ponto de Murcha Permanente (m³/m³)
    thFC=0.434,  # Capacidade de Campo (m³/m³)
    thS=0.620,   # Porosidade Total (Saturação) (m³/m³)
    Ksat=47.2,   # Condutividade Hidráulica Saturada (mm/h)
    penetrability=100
)

custom_soil.add_layer(
    thickness=0.1,  # Espessura da camada em metros (10-20 cm)
    thWP=0.279,
    thFC=0.439,
    thS=0.618,
    Ksat=14.3,
    penetrability=100
)

custom_soil.add_layer(
    thickness=0.1,  # Espessura da camada em metros (20-30 cm)
    thWP=0.301,
    thFC=0.480,
    thS=0.618,
    Ksat=9.1,
    penetrability=100
)

custom_soil.add_layer(
    thickness=0.1,  # Espessura da camada em metros (30-40 cm)
    thWP=0.293,
    thFC=0.500,
    thS=0.611,
    Ksat=8.0,
    penetrability=100
)

crop = Crop(
    'custom',  # Nome da cultura
    CropType=1,  # Vegetal folhoso (1)
    PlantMethod=0,  # Transplantado (0)
    CalendarType=1,  # Dias de calendário (1)
    SwitchGDD=0,  # Não usar Growing Degree Days (0)
    planting_date='08/01',  # Data de plantio
    harvest_date='08/30',  # Data de colheita
    EmergenceCD=8,  # Dias até recuperação da planta após transplante
    MaxRootingCD=20,  # Dias até profundidade máxima das raízes
    SenescenceCD=29,  # Dias até senescência
    MaturityCD=30,  # Dias até maturidade
    HIstart=6,  # Início da formação de rendimento após transplante
    FloweringCD=0,  # Não há floração (-999)
    YldFormCD=7,  # Duração da formação de rendimento
    Tbase=10.0,  # Temperatura base para crescimento (°C)
    Tupp=30.0,  # Temperatura máxima para crescimento (°C)
    Zmin=0.10,  # Profundidade mínima efetiva das raízes (m)
    Zmax=0.30,  # Profundidade máxima efetiva das raízes (m)
    fshape_r=15,  # Fator de forma para expansão das raízes
    SxTopQ=0.06,  # Extração máxima de água pelas raízes na camada superior (m³/m³/dia)
    SxBotQ=0.012,  # Extração máxima de água pelas raízes na camada inferior (m³/m³/dia)
    CCx=0.85,  # Cobertura máxima do dossel (fração)
    CGC_CD=0.9,  # Coeficiente de crescimento do dossel (fração de cobertura por dia)
    CDC_CD = 0.08,  # Fração de declínio do dossel por dia (8%)
    Kcb=1.10,  # Coeficiente de cultivo com o dossel completo antes da senescência
    WP=1700,  # Produtividade de água normalizada para ET0 e CO2 (g/m²)
    WPy=100,  # Produtividade de água ajustada durante a formação do rendimento (% do WP)
    fsink=1,  # Desempenho da cultura sob CO2 elevado (%)
    HI0=85,  # Índice de colheita inicial (em %)
    fshape_w1=2.5,  # Fator de forma para estresse hídrico na expansão do dossel
    fshape_w2=3.0,  # Fator de forma para controle estomático
    fshape_w3=3.0,  # Fator de forma para senescência do dossel
    fshape_w4=1.0,  # Fator de forma para polinização
    p_up1=0.4,  # Limiar superior de esgotamento de água no solo que afeta a expansão do dossel
    p_up2=0.50,  # Limiar superior de esgotamento de água no solo que afeta o controle estomático
    p_up3=0.85,  # Limiar superior de esgotamento de água no solo que afeta a senescência do dossel
    p_up4=0.90,  # Limiar superior de esgotamento de água no solo que afeta a polinização
    Tmin_up=8,  # Temperatura mínima para polinização
    Tmax_up=40,  # Temperatura máxima para polinização
    Aer=5,  # Volume (%) abaixo da saturação em que o estresse por aeração começa a ocorrer
    LagAer=3,  # Dias de atraso antes que o estresse por aeração afete o crescimento
    PlantPop = 160000,
    SeedSize=15, 
    p_lo1=0.20,  # Limiar inferior de esgotamento de água no solo para efeitos no crescimento do dossel
    p_lo2=0.25,  # Limiar inferior de esgotamento de água no solo para efeitos no controle estomático
    p_lo3=0.30,  # Limiar inferior de esgotamento de água no solo para efeitos na senescência do dossel
    p_lo4=0.45,  # Limiar inferior de esgotamento de água no solo para efeitos na polinização
    Determinant = 1,
    HIstartCD = 15,
    YldWC = 90,
)

# Definir as condições iniciais de umidade do solo
initWC = InitialWaterContent(value=['FC'])

# Definir o ambiente AquaCrop
class AquaCropEnv:
    def __init__(self, sim_start, sim_end, df_climate, soil, crop, init_wc, max_irrigation, max_steps_per_episode=1000):
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.df_climate = df_climate
        self.soil = soil
        self.crop = crop
        self.init_wc = init_wc
        self.max_irrigation = max_irrigation
        self.max_steps_per_episode = max_steps_per_episode
        self.wdf = self.calc_eto_faopm(df_climate)
        self.current_step = 0
        self.state = None
        self.done = False
        self.previous_biomass = 0
        self.current_day = sim_start
        self.action_space = spaces.Discrete(11)  # 11 ações: 0 a 25 mm de irrigação (2.5 mm por passo)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.state = np.random.randint(0, self.max_irrigation + 1)
        self.previous_biomass = 0  # Biomassa inicial
        self.done = False
        self.current_day = self.sim_start
        return self.get_obs()

    def step(self, action):
        if self.done:
            raise RuntimeError("Environment needs to be reset.")
        
        # Traduzir a ação em mm de irrigação
        irrigation_amount = self.action_to_irrigation(action)
        current_precipitation = self.wdf.loc[self.current_step, 'Precipitation']
        ETc = np.mean(self.wdf['eto'])  # Evapotranspiração de referência (ETO)

        # Cálculo do déficit hídrico (ETc - Precipitação)
        et_deficit = max(ETc - current_precipitation, 0)

        # Ajustar o gerenciamento de irrigação para o dia atual
        irr_mngt = IrrigationManagement(irrigation_method=1, SMT=[irrigation_amount] * 4)

        # Rodar o modelo AquaCrop para a temporada completa
        model = AquaCropModel(
            self.sim_start, self.sim_end, self.wdf, self.soil, self.crop,
            initial_water_content=self.init_wc, irrigation_management=irr_mngt
        )
        model.run_model(till_termination=True)

        # Acessar a biomassa diária
        try:
            biomassa_diaria = model._outputs.crop_growth['biomass']
        except KeyError:
            print("Erro: Coluna 'biomass' não encontrada!")
            biomassa_diaria = None

        # Recompensa baseada no déficit hídrico
        if irrigation_amount <= et_deficit:
            reward = 10  # Recompensa positiva se a irrigação estiver de acordo com o déficit
        elif irrigation_amount > et_deficit:
            reward = -abs(irrigation_amount - et_deficit)  # Penalidade por irrigar mais do que o necessário

        # Penalidade para subirrigação e superirrigação
        soil_moisture = self.get_soil_moisture()
        MAD = 0.2  # Defina o limite inferior para a umidade (Depleção Permitida)
        FC = 0.434  # Defina a capacidade de campo

        if soil_moisture < MAD:
            reward -= abs(soil_moisture - MAD)  # Penalidade por umidade muito baixa (subirrigação)
        elif soil_moisture > FC:
            reward -= abs(soil_moisture - FC)  # Penalidade por umidade muito alta (superirrigação)

        # Atualizar o estado
        new_state = (self.state + irrigation_amount) % self.max_irrigation
        self.current_step += 1

        # Verificar se atingimos o número máximo de dias
        self.done = self.current_step >= self.max_steps_per_episode

        # Exibir o estado, ação, irrigação e recompensa
        print(f"Step: {self.current_step}, State: {self.state}, Action: {action} "
              f"(Irrigation: {irrigation_amount} mm), Reward: {reward:.2f}")

        self.state = new_state
        return self.get_obs(), reward, self.done, {}

    def action_to_irrigation(self, action):
        return action * 2.5

    def calc_eto_faopm(self, df):
        df['eto'] = df['ReferenceET'].clip(0.1)
        return df

    def get_obs(self):
        state = [
            self.state,
            self.current_step,
            np.mean(self.wdf['MinTemp']),
            np.mean(self.wdf['MaxTemp']),
            np.mean(self.wdf['Precipitation']),
            np.mean(self.wdf['eto']),
            np.sum(self.wdf['Precipitation']),
            np.sum(self.wdf['eto'])
        ]
        return np.array(state, dtype=np.float32)

    def get_soil_moisture(self):
        # Exemplo de função que retorna a umidade do solo. Pode ser ajustada com base nos seus dados
        return np.random.uniform(0.1, 0.5)

# Criar o ambiente AquaCrop para o treinamento com os dados climáticos históricos
env = AquaCropEnv(sim_start, sim_end, df_climate_train, custom_soil, crop, initWC, max_irrigation=20)

# Definir o agente de Q-Learning
class QLearningAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.9, epsilon_greedy=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            q_vals = self.q_table[tuple(state)]
            action = np.argmax(q_vals)
        return action

    def learn(self, transition):
        s, a, r, next_s, done = transition
        q_val = self.q_table[tuple(s)][a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[tuple(next_s)])
        self.q_table[tuple(s)][a] += self.lr * (q_target - q_val)
        self.adjust_epsilon()

    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Função para executar o treinamento do agente
def run_qlearning(agent, env, num_episodes=1000, min_steps=50, max_steps=100):
    history = []
    for episode in range(num_episodes):
        state = env.reset()
        final_reward, n_moves = 0.0, 0
        max_steps_per_episode = np.random.randint(min_steps, max_steps + 1)
        env.max_steps_per_episode = max_steps_per_episode
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(Transition(state, action, reward, next_state, done))
            state = next_state
            n_moves += 1
            if done:
                break
            final_reward += reward
        print(f'Episode {episode + 1}: Final Reward: {final_reward:.2f}, Moves: {n_moves}')
        history.append((n_moves, final_reward))
    return history

# Definir a transição do agente
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Criar o agente
agent = QLearningAgent(env)

# Treinamento do agente
history = run_qlearning(agent, env)

# Função para prever a irrigação a partir dos dados climáticos
def get_state_from_climate_data(min_temp, max_temp, precipitation, reference_et):
    state = [
        0,
        0,
        min_temp,
        max_temp,
        precipitation,
        reference_et,
        precipitation,
        reference_et
    ]
    return np.array(state, dtype=np.float32)

# Função para carregar o arquivo Excel e realizar previsões

def process_uploaded_file(agent, file_path):
    try:
        df_climate_test = pd.read_excel(file_path)
        df_climate_test['Date'] = pd.to_datetime(df_climate_test['Date'], format='%Y-%m-%d')

        print(f"\nPrevisões de irrigação para o arquivo: {file_path}")
        for index, row in df_climate_test.iterrows():
            min_temp = row['MinTemp']
            max_temp = row['MaxTemp']
            precipitation = row['Precipitation']
            reference_et = row['ReferenceET']

            # Prever a quantidade de irrigação para o dia atual
            irrigation_amount = predict_irrigation(agent, min_temp, max_temp, precipitation, reference_et)
            print(f"Data: {row['Date']} - Irrigação recomendada: {irrigation_amount} mm")
    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")

# Função para prever a irrigação

def predict_irrigation(agent, min_temp, max_temp, precipitation, reference_et):
    state = get_state_from_climate_data(min_temp, max_temp, precipitation, reference_et)
    state_tuple = tuple(state)

    if state_tuple in agent.q_table:
        action = np.argmax(agent.q_table[state_tuple])
    else:
        action = np.random.choice(agent.env.action_space.n)

    irrigation_amount = agent.env.action_to_irrigation(action)
    return irrigation_amount

# Função para obter o estado a partir dos dados climáticos
def get_state_from_climate_data(min_temp, max_temp, precipitation, reference_et):
    state = [
        0,  # Estado inicial
        0,  # Passos iniciais
        min_temp,
        max_temp,
        precipitation,
        reference_et,
        precipitation,
        reference_et
    ]
    return np.array(state, dtype=np.float32)

# Interface para uploads contínuos
print("\nBem-vindo ao sistema de previsão de irrigação!")
while True:
    file_path = input("Digite o caminho do arquivo Excel para análise (ou 'sair' para encerrar): ")
    if file_path.lower() == 'sair':
        print("Encerrando o sistema. Até breve!")
        break

    if os.path.exists(file_path) and file_path.endswith('.xlsx'):
        process_uploaded_file(agent, file_path)
    else:
        print("Arquivo inválido ou não encontrado. Tente novamente.")
