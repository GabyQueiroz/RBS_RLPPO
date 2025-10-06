import os
os.environ['DEVELOPMENT'] = 'True'
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de exibição do pandas para mostrar todas as colunas e linhas
pd.set_option('display.max_columns', None)  # Mostrar todas as colunas
pd.set_option('display.max_rows', None)     # Mostrar todas as linhas
pd.set_option('display.expand_frame_repr', False)  # Não quebrar colunas no display


# Carregar o arquivo Excel com os dados climáticos
file_path = 'dados_DV.xlsx'  # Arquivo de clima fornecido
df_climate = pd.read_excel(file_path)

# Definindo as datas de início e fim da simulação
sim_start = '2007/06/01'
sim_end = '2007/12/31'

# Filtrar os dados climáticos para incluir apenas a época de cultivo
df_climate = df_climate[(df_climate['Date'] >= sim_start) & (df_climate['Date'] <= sim_end)]

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

# Inicializar o conteúdo de água no solo (Capacidade de Campo)
fc = InitialWaterContent(value=['FC'])

# Lista completa de ITN com 193 valores
irrigation_data = pd.DataFrame({
    'Date': pd.date_range(start='2007-06-01', periods=30),  
    'ITN': [
    13.25, 0, 0, 0, 0, 5, 5, 5, 0, 0, 
    0, 0, 0, 5, 5, 5, 0, 0, 5, 5, 
    5, 0, 0, 5, 0, 0, 0, 0, 0, 5
]})

print(irrigation_data)


# Criar o cronograma de irrigação com as datas e valores de ITN
schedule = pd.DataFrame({
    'Date': irrigation_data['Date'],
    'Depth': irrigation_data['ITN']  # Usar a coluna ITN para definir a profundidade de irrigação diária
})

# Verificar o conteúdo do cronograma de irrigação
print(schedule.head())  # Verificar se os valores de ITN estão corretos

# Criar o gerenciamento de irrigação com o cronograma
# Tentar outros métodos de irrigação
irrigation_management = IrrigationManagement(irrigation_method=3, Schedule=schedule)


# Configurar e executar o modelo AquaCrop com o cronograma de irrigação baseado nas regras
model = AquaCropModel(
    sim_start_time=sim_start,
    sim_end_time=sim_end,
    weather_df=df_climate,
    soil=custom_soil,
    crop=crop,
    irrigation_management=irrigation_management,
    initial_water_content=fc
)

# Rodar o modelo
model.run_model(till_termination=True)

# Extrair e analisar os resultados
results = model._outputs.final_stats[['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']]
print(model._outputs.final_stats[['Fresh yield (tonne/ha)', 'Yield potential (tonne/ha)', 'Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']])

# Criar figura com dois gráficos (rendimento e irrigação sazonal)
fig, ax = plt.subplots(2, 1, figsize=(10, 14))

# Gráfico de rendimento
sns.boxplot(data=results, y='Dry yield (tonne/ha)', ax=ax[0])
ax[0].set_title('Rendimento da Cultura (t/ha)', fontsize=18)
ax[0].set_ylabel('Rendimento Seco (t/ha)', fontsize=16)

# Gráfico de irrigação sazonal
sns.boxplot(data=results, y='Seasonal irrigation (mm)', ax=ax[1])
ax[1].set_title('Irrigação Sazonal Total (mm)', fontsize=18)
ax[1].set_ylabel('Irrigação (mm)', fontsize=16)

plt.tight_layout()
plt.show()
