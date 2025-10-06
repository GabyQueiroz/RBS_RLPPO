import pandas as pd
from datetime import datetime
import rule_engine

# Função para calcular o valor de Z
def calcular_Z(cultura):
    valores_Z = {
        "Abacaxi": 0.8,
        "Abobora": 0.9,
        "Alface": 0.7,
    }
    return valores_Z.get(cultura.lower().capitalize(), 0.0)

# Função para calcular DTA com base na textura do solo
def calcular_DTA(textura_solo):
    valores_DTA = {
        "Arenoso": 120,
        "Siltoso": 160,
        "Argiloso": 200,
    }
    return valores_DTA.get(textura_solo.lower().capitalize(), 0.0)

# Função para calcular f (fator de ajuste)
def calcular_f(cultura):
    fatores_f = {
        "Abacaxi": 0.9,
        "Abobora": 1.0,
        "Alface": 0.8,
    }
    return fatores_f.get(cultura.lower().capitalize(), 0.0)

# Função para obter fases de kc
def obter_fases_kc(cultura):
    fases_kc = {
        "Abacaxi": [0.7, 1.1, 0.9, 0.7],
        "Abobora": [0.9, 1.1, 0.7, 0.7],
        "Alface": [27, 37, 26, 10, 0.7, 1, 0.95],
    }
    return fases_kc.get(cultura, [0, 0, 0, 0])


# Função para calcular ETo usando a fórmula de Hargreaves (simplificação)
def calculate_eto(temp_max, temp_min):
    T = (temp_max + temp_min) / 2  # Temperatura média
    Qo = 17.0  # Valor fixo de Qo para exemplo
    ETo = 0.0023 * Qo * ((temp_max - temp_min) ** 0.5) * (T + 17.8)
    return ETo

# Função para ler os dados do arquivo
def ler_dados_climaticos(arquivo='dados_clima.txt'):
    df = pd.read_csv(arquivo, delimiter="\t")
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
    df['Horário'] = pd.to_datetime(df['Horário'], format='%H:%M').dt.time
    return df

# Função para ler dados do sensor com tratamento de horários
def ler_dados_sensor(arquivo='dados_sensor.txt'):
    df = pd.read_csv(arquivo, delimiter="\t")
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')

    # Tratar valores na coluna "Horário" para garantir que estão no formato correto
    def tratar_horario(horario):
        try:
            # Remover valores extras como ":00" ou outros que não seguem o padrão
            return datetime.strptime(horario.strip(), "%H:%M").time()
        except ValueError:
            return None

    df['Horário'] = df['Horário'].apply(tratar_horario)
    
    # Converter as colunas numéricas para float
    df['Umidade do Solo'] = pd.to_numeric(df['Umidade do Solo'], errors='coerce')
    df['Temperaturasolo'] = pd.to_numeric(df['Temperaturasolo'], errors='coerce')
    df['Temperatura do Ar'] = pd.to_numeric(df['Temperatura do Ar'], errors='coerce')
    df['Umidade Relativa'] = pd.to_numeric(df['Umidade Relativa'], errors='coerce')
    return df

# Função para obter a eficiência de irrigação
def obter_eficiencia_irrigacao(tipo_irrigacao):
    eficiencias = {
        "Sulcos": 65,
        "Faixas": 77.5,
        "Inundação": 85,
        "Convencional fixo": 75,
        "Convencional móvel": 70,
        "Carretel autopropelido": 65,
        "Pivo central linear móvel": 82.5,
        "Gotejamento": 82.5,
        "Microaspersão": 77.5,
    }
    return eficiencias.get(tipo_irrigacao.lower().capitalize(), 0.0)

# Função para calcular a lâmina de irrigação (Balanço Hídrico)
def calcular_lamina_irrigacao(forecast_data, cultura, ciclo_cultura, textura_solo, tipo_irrigacao):
    valor_Z = calcular_Z(cultura)
    valor_DTA = calcular_DTA(textura_solo)
    valor_f = calcular_f(cultura)
    fases_kc = obter_fases_kc(cultura)
    eficiencia_irrigacao = obter_eficiencia_irrigacao(tipo_irrigacao)
    
    # Agrupar os dados por dia para obter máximas e mínimas diárias e precipitação acumulada
    daily_summary = forecast_data.groupby('Data').agg(
        TempMax=('Temperatura', 'max'),
        TempMin=('Temperatura', 'min'),
        Precipitation=('Precipitação', 'sum')
    ).reset_index()

    cronograma_atual = []

    for index, row in daily_summary.iterrows():
        date_str = row['Data'].strftime('%Y/%m/%d')
        eto = calculate_eto(row['TempMax'], row['TempMin'])
        kc = fases_kc[4] if index == 0 else fases_kc[5]
        etc = eto * kc
        pe = row['Precipitation'] * valor_f
        etc_pe = etc - pe
        irn = max(0, etc_pe)  # Necessidade de irrigação líquida
        itn = irn / (eficiencia_irrigacao / 100)  # Necessidade de irrigação total

        cronograma_atual.append({
            'Dia': f'Dia {index + 1}',
            'date': date_str,
            'temp_max': row['TempMax'],
            'temp_min': row['TempMin'],
            'precipitation': row['Precipitation'],  # Precipitação acumulada diária
            'eto': eto,
            'Kc': kc,
            'ETc': etc,
            'Pe': pe,
            'ETc-Pe': etc_pe,
            'IRN': irn,
            'ITN': itn,
            'horario_irrigacao': "10:00"  # Definir o horário fixo para irrigação
        })

    return pd.DataFrame(cronograma_atual)

# Função para aplicar regras de irrigação e decisão usando rule_engine
def aplicar_regras(dataframe, forecast_data, sensor_data):
    rules = [
        rule_engine.Rule('ITN < 5'),  # Não irrigar se ITN for menor que 5mm
        rule_engine.Rule('ITN > 40'),  # Ajustar para máximo de 40mm se ITN for maior que 40mm
    ]

    decisions = []

    # Criar uma coluna para armazenar o horário da última atualização do sensor
    if 'Horário Atualização' not in sensor_data.columns:
        sensor_data['Horário Atualização'] = None

    for _, row in dataframe.iterrows():
        irrigar = False
        acoes = []
        irrigation_volume = row['ITN']  # Manter o valor calculado para "Irrigação Calculada"

        # Ajustar o dicionário de entrada para que o rule_engine possa acessar a coluna corretamente
        sensor_dict = row.to_dict()

        # Garantir que a "Temperatura" seja incluída no sensor_dict para ser verificada nas regras
        sensor_dict["Temperatura"] = float(forecast_data.loc[
            (forecast_data['Data'] == pd.to_datetime(row['date'], format='%Y/%m/%d')) &
            (forecast_data['Horário'] == datetime.strptime("10:00", "%H:%M").time()), 'Temperatura'].values[0]
        )

        sensor_dict["Temp_Solo"] = float(sensor_data.loc[sensor_data['Data'] == pd.to_datetime(row['date'], format='%Y/%m/%d'), 'Temperaturasolo'].values[0])

        # Verificar as regras para ITN (volume de irrigação calculado)
        for rule in rules:
            if rule.matches(sensor_dict):
                if rule.text == 'ITN < 5':
                    irrigation_volume = 0  # Definir volume de irrigação como 0
                    acoes.append("Não irrigar porque a irrigação calculada é <5mm")
                elif rule.text == 'ITN > 40':
                    irrigation_volume = 40  # Ajustar para máximo de 40mm
                    acoes.append("Lâmina de irrigação ajustada para máximo de 40mm")
                elif rule.text == 'Temp_Solo > 30':
                    irrigation_volume = 5  # Irrigar 5 mm
                    acoes.append("Irrigar 5 mm devido à Temperatura do Solo > 30°C")
                elif rule.text == 'Temperatura >= 32':
                    # Irrigar 5 mm se a temperatura for maior ou igual a 32°C
                    irrigation_volume = 5
                    acoes.append("Irrigar 5 mm devido à Temperatura >= 32°C ao longo do dia")

        # Checar se há horários ao longo do dia em que a temperatura seja >= 32°C e adicionar irrigação de 5mm
        horarios_quentes = forecast_data[
            (forecast_data['Data'] == pd.to_datetime(row['date'], format='%Y/%m/%d')) &
            (forecast_data['Temperatura'] >= 32)
        ]

        for _, horario in horarios_quentes.iterrows():
            # Não adicionar a irrigação de 5mm para o horário fixo de 10h
            if horario['Horário'] != datetime.strptime("10:00", "%H:%M").time():
                acoes.append(f"Irrigar 5mm, temperatura maior que 32°C no horário {horario['Horário']}")

                decisions.append({
                    'Dia': row['Dia'],
                    'date': row['date'],
                    'temp_max': row['temp_max'],
                    'temp_min': row['temp_min'],
                    'precipitation': row['precipitation'],
                    'eto': row['eto'],
                    'ETc': row['ETc'],
                    'irrigation_volume': 5,
                    'irrigacao_calculada': row['ITN'],
                    'horario_irrigacao': horario['Horário'].strftime('%H:%M'),
                    'acao': [f"Irrigar 5mm, temperatura maior que 32°C"]
                })

                # Atualizar os dados do sensor com base na irrigação de 5mm
                cond = (sensor_data['Data'] == pd.to_datetime(row['date'], format='%Y/%m/%d')) & (sensor_data['Horário'] == horario['Horário'])
                if not sensor_data[cond].empty:
                    sensor_data.loc[cond, 'Umidade do Solo'] += 5 * 0.1
                    sensor_data.loc[cond, 'Temperaturasolo'] -= 5 * 0.05
                    sensor_data.loc[cond, 'Temperatura do Ar'] -= 5 * 0.02
                    sensor_data.loc[cond, 'Umidade Relativa'] += 5 * 0.1
                    sensor_data.loc[cond, 'Horário Atualização'] = horario['Horário'].strftime('%H:%M')

        # Se ITN for entre 5 e 40 mm, irrigar o valor calculado às 10h
        if 5 <= irrigation_volume <= 40:
            acoes.append(f"Irrigar {irrigation_volume} mm de acordo com o cálculo")

        # Atualizar os dados do sensor apenas no horário fixo das 10h
        horario_atual = row['horario_irrigacao']  # Horário da irrigação

        # Filtrar a linha do sensor com a data e horário exato
        cond = (sensor_data['Data'] == pd.to_datetime(row['date'], format='%Y/%m/%d')) & (sensor_data['Horário'] == datetime.strptime(horario_atual, '%H:%M').time())
        if not sensor_data[cond].empty and irrigation_volume > 0:
            sensor_data.loc[cond, 'Umidade do Solo'] += irrigation_volume * 0.1
            sensor_data.loc[cond, 'Temperaturasolo'] -= irrigation_volume * 0.05
            sensor_data.loc[cond, 'Temperatura do Ar'] -= irrigation_volume * 0.02
            sensor_data.loc[cond, 'Umidade Relativa'] += irrigation_volume * 0.1
            sensor_data.loc[cond, 'Horário Atualização'] = horario_atual

        # Adicionar a irrigação fixa das 10h baseada na irrigação calculada
        decisions.append({
            'Dia': row['Dia'],
            'date': row['date'],
            'temp_max': row['temp_max'],
            'temp_min': row['temp_min'],
            'precipitation': row['precipitation'],
            'eto': row['eto'],
            'ETc': row['ETc'],
            'irrigation_volume': irrigation_volume,
            'irrigacao_calculada': row['ITN'],
            'horario_irrigacao': "10:00",
            'acao': [f"Irrigar {irrigation_volume} mm de acordo com o cálculo"]
        })

    return pd.DataFrame(decisions), sensor_data

# Função principal para executar o sistema
def executar_sistema():
    cultura = "Alface"
    ciclo_cultura = 30
    textura_solo = "Arenoso"
    tipo_irrigacao = "Gotejamento"

    print(f"Cultura: {cultura}")
    print(f"Ciclo da Cultura (dias): {ciclo_cultura}")
    print(f"Textura do Solo: {textura_solo}")
    print(f"Tipo de Irrigação: {tipo_irrigacao}")
    
    # Ler os dados climáticos e do sensor
    forecast_data = ler_dados_climaticos()
    sensor_data = ler_dados_sensor()

    # Calcular a lâmina de irrigação diária
    daily_df = calcular_lamina_irrigacao(forecast_data, cultura, ciclo_cultura, textura_solo, tipo_irrigacao)
    
    # Aplicar regras de irrigação e atualizar os dados do sensor
    decisions, updated_sensor_data = aplicar_regras(daily_df, forecast_data, sensor_data)

    # Gerar resumo diário
    resumo_diario = daily_df.groupby('date').agg(
        TempMax=('temp_max', 'max'),
        TempMin=('temp_min', 'min'),
        Precipitation=('precipitation', 'sum'),
        Irrigacao_Calculada=('ITN', 'mean'),
        Irrigacao_Total=('ITN', 'sum')
    ).reset_index()

    print("\nResumo Diário de Irrigação:")
    print(resumo_diario)

    print("\nTabela de Decisões de Irrigação:")
    print(decisions)

    # Sobrescrever o arquivo 'dados_sensor.txt' com os dados atualizados
    updated_sensor_data.to_csv('dados_sensor.txt', sep="\t", index=False)
    print("\nDados do sensor atualizados foram salvos no arquivo 'dados_sensor.txt'.")

# Iniciar a execução
executar_sistema()
