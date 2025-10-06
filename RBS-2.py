import os
os.environ['DEVELOPMENT'] = 'True'
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aquacrop.utils import prepare_weather, get_filepath

# Configurações para o pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Carregar o arquivo Excel com os dados climáticos
file_path = 'dados_DV.xlsx'  # Arquivo de clima fornecido
wdf = pd.read_excel(file_path)

# Definição de datas de simulação
sim_start = '2017/05/01'
sim_end = '2017/05/30'

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
    planting_date='05/01',  # Data de plantio
    harvest_date='05/30',  # Data de colheita
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
    SxTopQ=0.06,  # Extração máxima pelas raízes na camada superior (m³/m³/dia)
    SxBotQ=0.012,  # Extração máxima de água pelas raízes na camada inferior (m³/m³/dia)
    CCx=0.85,  # Cobertura máxima do dossel (fração)
    CGC_CD=0.9,  # Coeficiente de crescimento do dossel (fração de cobertura por dia)
    CDC_CD=0.08,  # Fração de declínio do dossel por dia (8%)
    Kcb=1.10,  # Coeficiente de cultivo com o dossel completo antes da senescência
    WP=1700,  # Produtividade de água normalizada para ET0 e CO2 (g/m²)
    WPy=100,  # Produtividade de água ajustada durante a formação do rendimento (% do WP)
    fsink=1,  # Desempenho da cultura sob CO2 elevado (% do WP)
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
    PlantPop=160000,
    SeedSize=15,
    p_lo1=0.20,  # Limiar inferior de esgotamento de água no solo para efeitos no crescimento do dossel
    p_lo2=0.25,  # Limiar inferior de esgotamento de água no solo para efeitos no controle estomático
    p_lo3=0.30,  # Limiar inferior de esgotamento de água no solo para efeitos na senescência do dossel
    p_lo4=0.45,  # Limiar inferior de esgotamento de água no solo para efeitos na polinização
    Determinant=1,
    HIstartCD=15,
    YldWC=90,
)

initWC = InitialWaterContent(wc_type='Pct', value=[40])
smts = [65, 55, 45, 45]
max_irr_season = 1000

irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrrSeason=max_irr_season)

# Inicialização do modelo AquaCrop
model = AquaCropModel(sim_start, sim_end, wdf, custom_soil, crop, irrigation_management=irrmngt, initial_water_content=initWC)

# Executar o modelo
model.run_model(till_termination=True)

# Saída de fluxos de água
water_out_df = pd.DataFrame(model._outputs.water_flux)

# Saída de armazenamento de água no solo
water_storage_df = pd.DataFrame(model._outputs.water_storage)

# Configuração dos parâmetros do solo (ajustar conforme necessário)
fc = 0.35  # Capacidade de campo (fração)
pwp = 0.20  # Ponto de murcha permanente (fração)

# Calcular a umidade média do solo e transformá-la em percentual (0-100%)
soil_moisture_columns = [f"th{i}" for i in range(1, 13)]
water_storage_df["SoilMoisturePercent"] = water_storage_df[soil_moisture_columns].mean(axis=1)*100

# Filtrar os dados climáticos para as datas da simulação
start_date = pd.to_datetime(sim_start)
end_date = pd.to_datetime(sim_end)
weather_df = pd.DataFrame({
    "Date": pd.to_datetime(wdf["Date"]),
    "MinTemp": wdf["MinTemp"],
    "MaxTemp": wdf["MaxTemp"],
    "Precipitation": wdf["Precipitation"],
    "ReferenceET": wdf["ReferenceET"],
})

# Filtrar os dados climáticos apenas para as datas simuladas
weather_simulated = weather_df[(weather_df["Date"] >= start_date) & (weather_df["Date"] <= end_date)].reset_index(drop=True)

# Adicionar a coluna "dap" ao DataFrame principal
weather_simulated["dap"] = range(1, len(weather_simulated) + 1)

# Adicionar a umidade do solo ao DataFrame principal
weather_simulated["SoilMoisturePercent"] = water_storage_df["SoilMoisturePercent"].values

# Função para calcular a irrigação com base nas fases e limites
def calculate_irrigation(dap, soil_moisture):
    # Definir o limite ideal para cada fase
    if dap <= 8:  # Fase 1
        threshold = 65
    elif 9 <= dap <= 19:  # Fase 2
        threshold = 55
    elif 20 <= dap <= 27:  # Fase 3
        threshold = 45
    elif 28 <= dap <= 30:  # Fase 4
        threshold = 45
    else:
        return 0  # Fora das fases definidas, sem irrigação

    # Cálculo de irrigação para atingir o limite ideal
    if soil_moisture < threshold:
        return (threshold - soil_moisture) * 1.5
    return 0

# Função para calcular a umidade após a irrigação considerando os limites de cada fase
def calculate_soil_moisture_after_irrigation(dap, soil_moisture, irrigation):
    # Definir o limite ideal para cada fase
    if dap <= 8:  # Fase 1
        threshold = 65
    elif 9 <= dap <= 19:  # Fase 2
        threshold = 55
    elif 20 <= dap <= 27:  # Fase 3
        threshold = 45
    elif 28 <= dap <= 30:  # Fase 4
        threshold = 45
    else:
        return soil_moisture  # Fora das fases definidas, retorna a umidade original

    # Atualizar a umidade após a irrigação
    updated_moisture = soil_moisture + (irrigation / 10)  # C9onversão mm -> %
    return min(updated_moisture, threshold)  # Garante que não excede o limite ideal

# Adicionar a coluna de irrigação ao DataFrame
weather_simulated["Irrigation"] = weather_simulated.apply(
    lambda row: calculate_irrigation(row["dap"], row["SoilMoisturePercent"]), axis=1
)

# Adicionar a coluna de umidade após a irrigação ao DataFrame
weather_simulated["SoilMoistureAfterIrrigation"] = weather_simulated.apply(
    lambda row: calculate_soil_moisture_after_irrigation(
        row["dap"], row["SoilMoisturePercent"], row["Irrigation"]
    ),
    axis=1
)

# Exibir os valores antes e depois da irrigação
print(weather_simulated[["Date", "dap", "SoilMoisturePercent", "Irrigation"]])
