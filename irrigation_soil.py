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

# Configuração do solo, cultura e manejo de irrigação
soil = Soil('ClayLoam')

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


initWC = InitialWaterContent(wc_type='Pct', value=[40])
smts = [75, 65, 55, 55]
max_irr_season = 1000
irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrrSeason=max_irr_season)

# Inicialização do modelo AquaCrop
model = AquaCropModel(sim_start, sim_end, wdf, soil, crop, irrigation_management=irrmngt, initial_water_content=initWC)

# Executar o modelo
model.run_model(till_termination=True)

# Saída de fluxos de água
water_out_df = pd.DataFrame(model._outputs.water_flux)

# Saída de armazenamento de água no solo
water_storage_df = pd.DataFrame(model._outputs.water_storage)

# Configuração dos parâmetros do solo (ajustar conforme necessário)
fc = 0.34  # Capacidade de campo (fração)
pwp = 0.20  # Ponto de murcha permanente (fração)

# Calcular a umidade média do solo e transformá-la em percentual (0-100%)
soil_moisture_columns = [f"th{i}" for i in range(1, 13)]
water_storage_df["SoilMoistureAvg"] = water_storage_df[soil_moisture_columns].mean(axis=1)
water_storage_df["SoilMoisturePercent"] = ((water_storage_df["SoilMoistureAvg"] - pwp) / (fc - pwp)) * 100
water_storage_df["SoilMoisturePercent"] = water_storage_df["SoilMoisturePercent"].clip(lower=0, upper=100)

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

# Adicionar a coluna de irrigação diária (IrrDay) ao DataFrame
weather_simulated["IrrDay"] = water_out_df["IrrDay"].values

# Adicionar a umidade do solo ao DataFrame principal
weather_simulated["SoilMoisturePercent"] = water_storage_df["SoilMoisturePercent"].values

# Exibir a tabela final
print(weather_simulated)

# (Opcional) Salvar os dados em um arquivo CSV
weather_simulated.to_csv("simulated_weather_irrigation.csv", index=False)

# Plotar a irrigação diária e a umidade do solo
plt.figure(figsize=(12, 6))

# Irrigação diária
plt.plot(weather_simulated["Date"], weather_simulated["IrrDay"], label="Irrigação Diária (mm)", color="blue")

# Umidade do solo
plt.plot(weather_simulated["Date"], weather_simulated["SoilMoisturePercent"], label="Umidade do Solo (%)", color="green")

plt.xlabel("Data", fontsize=12)
plt.ylabel("Valores", fontsize=12)
plt.title("Irrigação e Umidade do Solo ao Longo do Tempo", fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Acessar as estatísticas finais
final_stats = model._outputs.final_stats
# Acesse os atributos para análise mais detalhada
print(model._outputs.water_flux)  # Detalhes sobre fluxos de água


# Formatando e exibindo as estatísticas finais
print("\nEstatísticas Finais da Temporada:")
formatted_stats = {
    "Season": int(final_stats["Season"].iloc[0]),
    "Crop Type": final_stats["crop Type"].iloc[0],
    "Harvest Date (YYYY/MM/DD)": final_stats["Harvest Date (YYYY/MM/DD)"].iloc[0],
    "Harvest Date (Step)": int(final_stats["Harvest Date (Step)"].iloc[0]),
    "Dry yield (tonne/ha)": final_stats["Dry yield (tonne/ha)"].iloc[0],
    "Fresh yield (tonne/ha)": final_stats["Fresh yield (tonne/ha)"].iloc[0],
    "Yield potential (tonne/ha)": final_stats["Yield potential (tonne/ha)"].iloc[0],
    "Seasonal irrigation (mm)": final_stats["Seasonal irrigation (mm)"].iloc[0],
}

for key, value in formatted_stats.items():
    print(f"{key}: {value}")

# (Opcional) Salvar as estatísticas finais formatadas em um arquivo CSV
pd.DataFrame([formatted_stats]).to_csv("final_statistics.csv", index=False)
